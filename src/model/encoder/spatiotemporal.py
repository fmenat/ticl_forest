
import torch

from ..layers.advanced import ConvLSTM
from ..layers.simple import DoubleConv, Down, NewDown, NewDoubleConv, TempCNN

class TDConv_ConvLSTM(torch.nn.Module):
    def __init__(self, n_feats, filters=[64, 128, 256], lstm_hidden=[256], new_layers=True, **encoder_args):
        super().__init__()
        self.n_feats = n_feats
        self.filters = filters
        self.n_layers = len(filters)
        self.lstm_hidden = lstm_hidden
        if new_layers:
            DownLayer = NewDown
            backboneConv = NewDoubleConv
        else:
            DownLayer = Down
            backboneConv = DoubleConv

        self.conv_encoder = []
        for n in range(self.n_layers):
            if n == 0:
                self.conv_encoder.append(backboneConv(self.n_feats, self.filters[0], **encoder_args["per_layer"][0] if "per_layer" in encoder_args else {}, **encoder_args))
            else:
                self.conv_encoder.append(DownLayer(self.filters[n-1], self.filters[n],  **encoder_args["per_layer"][n] if "per_layer" in encoder_args else {}, **encoder_args))
        self.conv_encoder =  torch.nn.ModuleList(self.conv_encoder)

        self.convlstm = ConvLSTM(self.filters[-1], self.lstm_hidden)

    def get_channels_out(self):
        return self.lstm_hidden[-1]
    
    def forward(self, x: torch.Tensor): #expected shape [B, T, C, H, W]
        #print("Input shape:", x.shape)
        encoder_values = [] 
        for n in range(self.n_layers):
            encoder_values_t = []
            for t in range(x.shape[1]): #each time/frame is encoded (i.e. Time-distributed CNN)
                encoder_ =  self.conv_encoder[n](encoder_values[-1][:, t] if n > 0 else x[:, t])
                encoder_values_t.append(encoder_)
             # Stack time dimension for ConvLSTM
            encoder_values.append(torch.stack(encoder_values_t, dim=1)) # [B, T, C, H/n**2, W/n**2
            #print(f"Encoder layer {n} output shape:", encoder_values[-1].shape)
        
        if encoder_values[-1].shape[1] == 1:
            bottleneck = encoder_values[-1][:, 0]  # If only one time step, skip ConvLSTM
        else:
            bottleneck = self.convlstm(encoder_values[-1])     # [B, hidden, H/n**2, W/n**2]
        
        return { 
            "repre": bottleneck, # [B, hidden, H/n**2, W/n**2]
            "skip": encoder_values #enc values is a list of tensors for each layer, with shape [B, T, C, H/n**2, W/n**2]
            }
    


class TDConv_Simple(torch.nn.Module):
    def __init__(self, n_feats, filters=[64, 128, 256], temporal_aggregation = "sub", new_layers=True, **encoder_args):
        super().__init__()
        self.n_feats = n_feats
        self.filters = filters
        self.n_layers = len(filters)
        self.temporal_aggregation = temporal_aggregation
        if new_layers:
            DownLayer = NewDown
            backboneConv = NewDoubleConv
        else:
            DownLayer = Down
            backboneConv = DoubleConv

        self.conv_encoder = []
        for n in range(self.n_layers):
            if n == 0:
                self.conv_encoder.append(backboneConv(self.n_feats, self.filters[0], **encoder_args["per_layer"][0] if "per_layer" in encoder_args else {}, **encoder_args))
            else:
                self.conv_encoder.append(DownLayer(self.filters[n-1], self.filters[n], **encoder_args["per_layer"][n] if "per_layer" in encoder_args else {},**encoder_args))
        self.conv_encoder =  torch.nn.ModuleList(self.conv_encoder)

    def get_channels_out(self):
        return self.filters[-1] if self.temporal_aggregation != "concatenate" else self.filters[-1]*2
    
    def forward(self, x: torch.Tensor): #expected shape [B, T, C, H, W]
        #print("Input shape:", x.shape)
        encoder_values = [] 
        for n in range(self.n_layers):
            encoder_values_t = []
            for t in range(x.shape[1]): #each time/frame is encoded (i.e. Time-distributed CNN)
                encoder_ =  self.conv_encoder[n](encoder_values[-1][:, t] if n > 0 else x[:, t])
                encoder_values_t.append(encoder_)
             # Stack time dimension for ConvLSTM
            encoder_values.append(torch.stack(encoder_values_t, dim=1)) # [B, T, C, H/n**2, W/n**2
            #print(f"Encoder layer {n} output shape:", encoder_values[-1].shape)


        if self.temporal_aggregation == "subtract":
            bottleneck = encoder_values[-1][:, -1]  - encoder_values[-1][:, 0]  # Subtraction between last and first time step
        elif self.temporal_aggregation == "average":
            bottleneck = encoder_values[-1].mean(dim=1)  # Average over time
        elif self.temporal_aggregation == "maximum":
            bottleneck = encoder_values[-1].max(dim=1).values  # Max over time
        elif self.temporal_aggregation == "minimum": #### NEW
            bottleneck = encoder_values[-1].min(dim=1).values  # Min over time
        elif self.temporal_aggregation == "concatenate":
            bottleneck = encoder_values[-1].reshape(encoder_values[-1].shape[0], -1, *encoder_values[-1].shape[3:]) #torch.cat([encoder_values[-1][:, t] for t in range(encoder_values[-1].shape[1])], dim=1)  # Concatenate over channel dimension
        else:
            raise ValueError(f"Unknown temporal aggregation method: {self.temporal_aggregation}")
        #print("Bottleneck shape:", bottleneck.shape)
        return { 
            "repre": bottleneck, # [B, hidden, H/n**2, W/n**2]
            "skip": encoder_values #enc values is a list of tensors for each layer, with shape [B, T, C, H/n**2, W/n**2]
            }
    


class TDConv_1DConv(torch.nn.Module):
    def __init__(self, n_feats, filters=[64, 128, 256], temp_filters=[256], n_times=0, first="spatial", new_layers=True, **encoder_args):
        super().__init__()
        self.n_feats = n_feats
        self.filters = filters
        self.n_layers = len(filters)
        self.temp_filters = temp_filters
        self.n_times = n_times
        self.first = first
        if new_layers:
            DownLayer = NewDown
            backboneConv = NewDoubleConv
        else:
            DownLayer = Down
            backboneConv = DoubleConv

        self.conv_encoder = []
        for n in range(self.n_layers):
            if n == 0:
                self.conv_encoder.append(backboneConv(self.n_feats if self.first == "spatial" else self.temp_filters[-1], self.filters[0], **encoder_args["per_layer"][0] if "per_layer" in encoder_args else {}, **encoder_args))
            else:
                self.conv_encoder.append(DownLayer(self.filters[n-1], self.filters[n],  **encoder_args["per_layer"][n] if "per_layer" in encoder_args else {}, **encoder_args))
        self.conv_encoder =  torch.nn.ModuleList(self.conv_encoder)

        if self.first == "spatial":
            self.tempcnn = TempCNN(self.filters[-1]*self.n_times, self.temp_filters, **encoder_args)
        elif self.first=="temporal":
            self.tempcnn = TempCNN(self.n_feats*self.n_times, self.temp_filters, **encoder_args)
        
    def get_channels_out(self):
        if self.first == "spatial":
            return self.temp_filters[-1]
        elif self.first == "temporal":
            return self.filters[-1]        
    
    def forward(self, x: torch.Tensor): #expected shape [B, T, C, H, W]
        n_batch = x.shape[0]
        n_times = x.shape[1]
        #print("Input shape:", x.shape)

        if self.first == "temporal":
            if n_times != self.n_times:
                pad_size = self.n_times - n_times
                x = torch.nn.functional.pad(x, (0,0,0,0,0,0,0,pad_size), mode='constant', value=0)
            x = self.tempcnn(x.reshape(n_batch, -1, x.shape[3], x.shape[4])).unsqueeze(1)  # [B, 1, C, H, W]
            
        encoder_values = [] 
        for n in range(self.n_layers):
            encoder_values_t = []
            for t in range(x.shape[1]): #each time/frame is encoded (i.e. Time-distributed CNN)
                encoder_ =  self.conv_encoder[n](encoder_values[-1][:, t] if n > 0 else x[:, t])
                encoder_values_t.append(encoder_)
            # Stack time dimension for ConvLSTM
            encoder_values.append(torch.stack(encoder_values_t, dim=1)) # [B, T, C, H/n**2, W/n**2
            #print(f"Encoder layer {n} output shape:", encoder_values[-1].shape)
        
        if self.first == "spatial":
            if n_times != self.n_times:
                pad_size = self.n_times - n_times
                encoder_values[-1] = torch.nn.functional.pad(encoder_values[-1], (0,0,0,0,0,0,0,pad_size), mode='constant', value=0)
            bottleneck = self.tempcnn(encoder_values[-1].reshape(n_batch, -1, *encoder_values[-1].shape[3:]))     # [B, hidden, H/n**2, W/n**2]
        
        elif self.first == "temporal":
            bottleneck = encoder_values[-1][:,0]     # [B, hidden, H/n**2, W/n**2]
        #print("Bottleneck shape:", bottleneck.shape)

        return { 
            "repre": bottleneck, # [B, hidden, H/n**2, W/n**2]
            "skip": encoder_values #enc values is a list of tensors for each layer, with shape [B, T, C, H/n**2, W/n**2]
            }
