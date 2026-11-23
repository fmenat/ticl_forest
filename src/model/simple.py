import torch

from .core import ForestMaskSegmentationBase
from .encoder.spatiotemporal import TDConv_ConvLSTM, TDConv_Simple, TDConv_1DConv
from .layers.simple import UnetUp
from .decoder.model import SegmentationHead


class UNetConvLSTM(ForestMaskSegmentationBase):
    def __init__(self, encoder_args, decoder_args, no_data_value=None, optim_args ={}, loss_args={},macro_loss_args={}):
        super().__init__(n_classes=decoder_args.get("n_classes", 2), no_data_value=no_data_value, 
                         label_name=decoder_args.get("label_name", ""), loss_args=loss_args, macro_loss_args=macro_loss_args, **optim_args)

        main_spatial = ""
        encoder = {}
        for k, v in encoder_args.items():
            if k in ["S2", "S1", "alphaearth", "landsat"]:
                main_spatial = k
                if v.get("type", "tdcnn") == "tdcnn":
                    encoder[k] = TDConv_ConvLSTM(**v)
                    encoder_args[k]["filters"]  = [v for k,v  in encoder[k].conv_encoder.channels.items()]
                elif v.get("type", "tdcnn") == "temp+spatcnn":
                    encoder[k] = TDConv_1DConv(**v)
        self.encoder = torch.nn.ModuleDict(encoder)

        emb_channels = self.encoder[main_spatial].get_channels_out()
        
        decoder = []
        for n in range(len(decoder_args["filters"])):
            decoder.append(UnetUp(emb_channels if n==0 else decoder_args["filters"][n-1], decoder_args["filters"][n], 
                                       skip_channels=encoder_args[main_spatial]["filters"][::-1][n+1], **decoder_args))
        self.decoder =  torch.nn.ModuleList(decoder)
        self.skipcon_mode = decoder_args.get("skipcon_mode", "last")
        if self.skipcon_mode == "weighted":
            self.temporal_weights = torch.nn.Parameter(torch.ones(encoder_args[main_spatial].get("n_times")))  #one weight per time step
        
        self.head = SegmentationHead(decoder_args["filters"][-1], decoder_args.get("n_classes", 2), emb_dims=decoder_args["filters"][-1], **decoder_args.get("head_args", {}))
        
    def forward_skip(self, enc_value, attn_mask=None):
        if self.skipcon_mode == "last":
            returned_value = enc_value[:, -1] #last time step

        elif self.skipcon_mode == "average":
            returned_value = enc_value.mean(dim=1) #average over time

        elif self.skipcon_mode == "maximum":
            returned_value = enc_value.max(dim=1).values  # max over time

        elif self.skipcon_mode == "weighted":
            weights = torch.nn.functional.softmax(self.temporal_weights, dim=0)
            times_ = enc_value.shape[1]
            returned_value = (enc_value * weights[None, -times_:, None, None, None]).sum(dim=1)  

        elif self.skipcon_mode == "attention":
            attn = attn_mask.mean(dim=0)  # average over heads -> BxTxHxW
            attn = torch.nn.Upsample(size=enc_value.shape[-2:], mode="bilinear", align_corners=False)(attn)
            returned_value = (enc_value * attn[:, :, None, :, :]).sum(dim=1)

        return returned_value
    
    def forward(self, batch_dict: dict): 
        # Each sensor data is encoded independently
        feat_sensors = {}
        for sensor in self.encoder: 
            if sensor in batch_dict:
                feat_sensors[sensor] = self.encoder[sensor](batch_dict[sensor])
            else:
                raise ValueError(f"Sensor {sensor} not found in input batch_dict")
            
        encoder_output = feat_sensors[sensor]  # Using only one encoder output for decoding

        if self.skipcon_mode == "attention":
            _, att_mask = self.attention_model(encoder_output["skip"][-1]) #tODO: maybe aggregation should also be done based on this attention
            encoder_output["att"] = att_mask
        else:
            att_mask = None

        # Decode (use last encoder states for skip connections) 
        dec_ = encoder_output["repre"]
        for n in range(len(self.decoder)):
            #print("Input to decoder layer", n, "shape:", encoder_output["repre"].shape if n == 0 else dec_.shape, "skip shape:", encoder_output["skip"][-(2+n)].shape)
            dec_ = self.decoder[n](dec_, self.forward_skip(encoder_output["skip"][-(2+n)], att_mask))
            #print("Output layer shape:", dec_.shape)
        logits = self.head(dec_)
        #print("Head logits shape:", logits.shape)
        
        return { "logits": self.prediction_trimming(logits), **encoder_output }
    

class UNetSimple(ForestMaskSegmentationBase):    
    # This is a simple UNet-like model without ConvLSTM, for ablation purposes    
    def __init__(self, encoder_args, decoder_args, no_data_value=None, optim_args ={}, loss_args={},macro_loss_args={}):
        super().__init__(n_classes=decoder_args.get("n_classes", 2), no_data_value=no_data_value, 
                         label_name=decoder_args.get("label_name", ""), loss_args=loss_args, macro_loss_args=macro_loss_args, **optim_args)
        
        main_spatial = ""
        encoder = {}
        for k, v in encoder_args.items():
            if k in ["S2", "S1", "alphaearth", "landsat"]:
                main_spatial = k
                if v.get("type", "standard") == "standard":
                    encoder[k] = TDConv_Simple(**v)
        self.encoder = torch.nn.ModuleDict(encoder)

        emb_channels = self.encoder[main_spatial].get_channels_out()
        
        decoder = []
        for n in range(len(decoder_args["filters"])):
            decoder.append(UnetUp(emb_channels if n==0 else decoder_args["filters"][n-1], decoder_args["filters"][n], 
                                       skip_channels=encoder_args[main_spatial]["filters"][::-1][n+1], **decoder_args)) 
        if len(decoder) == 0:  # If no decoder layers ==> identity layer
            decoder_args["filters"] = [emb_channels]

        self.decoder =  torch.nn.ModuleList(decoder)
        self.skipcon_mode = decoder_args.get("skipcon_mode", "last")
        if self.skipcon_mode == "weighted":
            self.temporal_weights = torch.nn.Parameter(torch.ones(encoder_args[main_spatial].get("n_times")))  #one weight per time step
        
        self.head = SegmentationHead(decoder_args["filters"][-1], decoder_args.get("n_classes", 2), emb_dims=decoder_args["filters"][-1], **decoder_args.get("head_args", {}))
        
    def forward_skip(self, enc_value, attn_mask=None):
        return UNetConvLSTM.forward_skip(self, enc_value, attn_mask=attn_mask)

    def forward(self, batch_dict: dict):
        return UNetConvLSTM.forward(self, batch_dict)


class EncoderDecoder(ForestMaskSegmentationBase):
    def __init__(self, encoder_args, decoder_args, no_data_value=None, optim_args ={}, loss_args={},macro_loss_args={}):
        super().__init__(n_classes=decoder_args.get("n_classes", 2), no_data_value=no_data_value, 
                         label_name=decoder_args.get("label_name", ""), loss_args=loss_args, macro_loss_args=macro_loss_args, **optim_args)
        
        main_spatial = ""
        encoder = {}
        for k, v in encoder_args.items():
            if k in ["S2", "S1", "alphaearth", "landsat"]:
                main_spatial = k
                if v.get("type", "tdcnn_convlstm") == "tdcnn_convlstm":
                    encoder[k] = TDConv_ConvLSTM(**v)
                elif v.get("type", "tdcnn_convlstm") == "tdcnn_simple":
                    encoder[k] = TDConv_Simple(**v)
                elif v.get("type", "tdcnn_convlstm") == "temp+spatcnn":
                    encoder[k] = TDConv_1DConv(**v)

        self.encoder = torch.nn.ModuleDict(encoder)

        emb_channels = self.encoder[main_spatial].get_channels_out()
        
        decoder = []
        for n in range(len(decoder_args["filters"])):
            decoder.append(UnetUp(emb_channels if n==0 else decoder_args["filters"][n-1], decoder_args["filters"][n], 
                                       skip_merge=None, **decoder_args))
        if len(decoder) == 0:  # If no decoder layers inform decoder
            decoder_args = decoder_args.copy()  # avoid modifying original decoder_args
            decoder_args["filters"] = [emb_channels]
        self.decoder =  torch.nn.ModuleList(decoder)
        
        self.head = SegmentationHead(decoder_args["filters"][-1], decoder_args.get("n_classes", 2), emb_dims=decoder_args["filters"][-1], **decoder_args.get("head_args", {}))
        
    
    def forward(self, batch_dict: dict): 
        # Each sensor data is encoded independently
        feat_sensors = {}
        for sensor in self.encoder: 
            if sensor in batch_dict:
                feat_sensors[sensor] = self.encoder[sensor](batch_dict[sensor])
            else:
                raise ValueError(f"Sensor {sensor} not found in input batch_dict")
            
        encoder_output = feat_sensors[sensor]  # Using only one encoder output for decoding

        # Decode (use last encoder states for skip connections) 
        dec_ = encoder_output["repre"]
        for n in range(len(self.decoder)):
            dec_ = self.decoder[n](dec_, None)
        logits = self.head(dec_)
        
        return { "logits": self.prediction_trimming(logits), **encoder_output }