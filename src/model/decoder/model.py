import torch

from ..layers.simple import DoubleConv
from ..encoder.vector import MLP

class SegmentationHead(torch.nn.Module):
    def __init__(self, in_channels, out_channels, emb_dims=-1, new_layers=True, **kwargs):
        super().__init__()
        self.new_layers = new_layers
        if self.new_layers:
            self.conv_layers = DoubleConv(in_channels, emb_dims if emb_dims > 0 else out_channels, **kwargs)
            self.linear = torch.nn.Conv2d(emb_dims if emb_dims > 0 else out_channels, out_channels, kernel_size=1) #1x1 conv
        else:
            self.linear = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1) #1x1 conv
        
    def forward(self, enc_feat):
        if self.new_layers:
            enc_feat = self.conv_layers(enc_feat)
        return self.linear(enc_feat)
