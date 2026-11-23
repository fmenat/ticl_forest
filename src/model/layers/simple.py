import numpy as np
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0, dilation=1, norm="batch", act="leakyrelu", **kwargs):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding="same", dilation=dilation),
            nn.BatchNorm2d(out_channels) if norm == "batch" else nn.GroupNorm(1, out_channels), 
            nn.LeakyReLU(inplace=True) if act == "leakyrelu" else nn.GELU(approximate='tanh'),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding="same", dilation=dilation),
            nn.BatchNorm2d(out_channels) if norm == "batch" else nn.GroupNorm(1, out_channels), 
            nn.LeakyReLU(inplace=True) if act == "leakyrelu" else nn.GELU(approximate='tanh'),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Dropout(kwargs["dropout_any"]) if kwargs.get("dropout_any",0) > 0 else nn.Identity()
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels, pool=True, **kwargs):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2) if pool else nn.Identity(),
            DoubleConv(in_channels, out_channels, **kwargs)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class NewDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0, dilation=1, norm="batch", act="leakyrelu", **kwargs):
        super().__init__()
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding="same", dilation=dilation),
            nn.BatchNorm2d(out_channels) if norm == "batch" else nn.GroupNorm(1, out_channels), 
            nn.LeakyReLU(inplace=True) if act == "leakyrelu" else nn.GELU(approximate='tanh'),
        )
        self.convblock2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size, padding="same", dilation=dilation),
            nn.BatchNorm2d(out_channels) if norm == "batch" else nn.GroupNorm(1, out_channels), 
            nn.LeakyReLU(inplace=True) if act == "leakyrelu" else nn.GELU(approximate='tanh'),
        )
        if dropout > 0:
            self.drop = nn.Dropout2d(dropout)
        elif kwargs.get("dropout_any",0) > 0:
            self.drop = nn.Dropout(kwargs["dropout_any"]) 
        else:
            self.drop = nn.Identity()

    def forward(self, x):
        first = self.convblock1(x)
        out = self.convblock2(first) #+ first  # Residual connection as UTAE
        out = self.drop(out)
        return out + first #residual conection as Ferrari

class NewDown(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels, pool=True, **kwargs):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1) if pool else nn.Identity(),
            NewDoubleConv(in_channels, out_channels, **kwargs)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UnetUp(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, skip_merge='concat', skip_channels=0, skip_attention=False, upscale_flag=True, new_layers=False, norm="batch", act="leakyrelu", **kwargs):
        super().__init__()
        self.skip_merge = skip_merge
        self.skip_attention = skip_attention
        self.upscale_flag = upscale_flag
        if self.upscale_flag:
            self.up_layer = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        if self.skip_attention:
            self.skip_layer = nn.Sequential(
                nn.Conv2d(skip_channels, in_channels if skip_merge == 'add' else skip_channels, kernel_size=1),
                SCSEModule(in_channels if skip_merge == 'add' else skip_channels, act=act),
            )
        else:
            self.skip_layer = nn.Sequential(
                nn.Conv2d(skip_channels, in_channels if skip_merge == 'add' else skip_channels, kernel_size=1),
                nn.BatchNorm2d(in_channels if skip_merge == 'add' else skip_channels) if norm == "batch" else nn.GroupNorm(1, in_channels if skip_merge == 'add' else skip_channels),
                nn.LeakyReLU(inplace=True) if act == "leakyrelu" else nn.GELU(approximate='tanh'),
            )
        if new_layers:
            self.conv = NewDoubleConv(in_channels + skip_channels if self.skip_merge =="concat" else in_channels, out_channels, norm=norm, act=act, **kwargs)
        else:
            self.conv = DoubleConv(in_channels + skip_channels if self.skip_merge =="concat" else in_channels, out_channels, norm=norm, act=act, **kwargs)
        
    def forward(self, enc_feat, skip_feat= None):
        if self.upscale_flag:
            enc_feat = self.up_layer(enc_feat)

        if skip_feat is not None:
            skip_feat = self.skip_layer(skip_feat)
        
            if self.skip_merge == 'concat':
                x = torch.cat([skip_feat, enc_feat], dim=1)
            elif self.skip_merge == 'add':
                x = skip_feat + enc_feat

        else:
            x = enc_feat

        return self.conv(x)


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=kernel_size // 2) #4 times because of 3 gates + cell
        self.hidden_dim = hidden_dim

    def forward(self, x, hidden):
        h_cur, c_cur = hidden
        combined = torch.cat([x, h_cur], dim=1)
        conv_out = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(conv_out, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, spatial_size, device):
        height, width = spatial_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=device))
    

class EmbeddingLayer(nn.Module):
    def __init__(self, in_features, out_features, model_type="linear", norm_layer=False):
        """
        Note: model_type: linear and conv1d with kernel size gives same result, linear is bit faster
            ref: https://stackoverflow.com/questions/55576314/conv1d-with-kernel-size-1-vs-linear-layer
        Parameters
        ----------
        in_features:
        out_features
        model_type
        """
        super().__init__()
        self.model_type = model_type
        if self.model_type == "linear":
            self.model = nn.Linear(in_features=in_features, out_features=out_features)
        elif self.model_type == "conv1d":
            self.model = nn.Conv1d(  # for conv 1d req shape: batch_size x in_features x seq_len
                in_features, out_features, 1
            )
        else:
            raise KeyError(f"Model type: {model_type} is not implemented for {self._get_name()} class")
        self.norm_layer = nn.LayerNorm(out_features) if norm_layer else nn.Identity()

    def forward(self, x):
        if self.model_type == "linear":
            x= self.model(x)
        elif self.model_type == "conv1d":
            x = self.model(x.permute(0, 2, 1)).permute(  # for conv 1d req shape: batch_size x in_features x seq_len
                0, 2, 1
            )  # we change output shape back to shape: batch_size x seq_len x in_features
        return self.norm_layer(x)
    


class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16, act="leakyrelu", **kwargs):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.LeakyReLU(inplace=True) if act == "leakyrelu" else nn.GELU(approximate='tanh'),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)
    

class TempCNN(nn.Module):
    def __init__(self, in_channels, out_channels, norm="batch", act="leakyrelu", **kwargs):
        super().__init__()
        layers = []
        for i, c in enumerate(out_channels):
            layers += [
                nn.Conv2d(in_channels if i == 0 else out_channels[i-1], c, 1, padding="same"),
                nn.BatchNorm2d(c) if norm == "batch" else nn.GroupNorm(1, c), 
                nn.LeakyReLU(inplace=True) if act == "leakyrelu" else nn.GELU(approximate='tanh'),
            ]
    
        self.tempcnn = nn.Sequential(*layers)

    def forward(self, x):
        return self.tempcnn(x)