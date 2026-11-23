import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import v2

from .utils import apply_indexing

class CustomScaler(torch.nn.Module):
    def __init__(self, mode, norm_data, eps=1e-8):
        super().__init__()
        self.mode = mode
        self.eps = eps
        self.norm_data = {band: torch.tensor(norm_data.sel(norm=band).values) for band in norm_data.coords["norm"].values}

    def forward(self, tensor: torch.Tensor):
        if self.mode == "standard":
            return (tensor - self.norm_data["mean"])/(self.norm_data["std"] + self.eps)

        elif self.mode == "minmax":
            return (tensor - self.norm_data["min"])/(self.norm_data["max"] - self.norm_data["min"] + self.eps)
        
        elif self.mode =="max":
            return tensor/(self.norm_data["max"] + self.eps)
        
        elif self.mode == "robust":
            return (tensor - self.norm_data["median"])/(self.norm_data["iqr"] + self.eps)

class MaskFillValue(torch.nn.Module):
    def __init__(self, mask_value=np.nan):
        super().__init__()
        self.mask_value = mask_value

    def forward(self, tensor: torch.Tensor):
        nan_mask = torch.isnan(tensor)
        if nan_mask.any():
            tensor = torch.where(nan_mask, self.mask_value, tensor)
        return tensor

class CatToOneHot(torch.nn.Module):
    def __init__(self, categories):
        super().__init__()
        self.categories = categories

    def forward(self, tensor: torch.Tensor):
        return torch.nn.functional.one_hot(tensor.squeeze(-1), num_classes=len(self.categories))

class RandGroupTransform(torch.nn.Module):
    def __init__(self, vertical_prob=0.0, horizontal_prob=0.0, rotate_prob=0.0, noise_prob=0.0, gaussianblur_prob=0.0, channeldrop_prob=0.0, crop_size=None):
        super().__init__()
        self.vertical_prob = vertical_prob
        self.horizontal_prob = horizontal_prob
        self.rotate_prob = rotate_prob
        self.noise_prob = noise_prob
        self.gaussianblur_prob = gaussianblur_prob
        self.channeldrop_prob = channeldrop_prob
        self.crop_size = crop_size

    def forward(self, input_tensor_dict, label_tensor_dict):
        if self.crop_size is not None:
            i, j, h, w = v2.RandomCrop(self.crop_size).get_params(label_tensor_dict["disturbances"][0], (self.crop_size, self.crop_size))

        aug = []
        if self.vertical_prob != 0 and torch.rand(1).item() < self.vertical_prob:
            aug.append("verticalflip")
        if self.horizontal_prob != 0 and torch.rand(1).item() < self.horizontal_prob:
            aug.append("horizontalflip")
        if self.rotate_prob != 0 and torch.rand(1).item() < self.rotate_prob:
            aug.append("rotate")
            n_rotations = torch.randint(4, size=(1,)).item()
        if self.noise_prob != 0 and torch.rand(1).item() < self.noise_prob:
            aug.append("noise")
        if self.gaussianblur_prob !=0 and torch.rand(1).item() < self.gaussianblur_prob:
            aug.append("gaussianblur")

        return_dic = {}
        for key, (tensor, dims) in dict(input_tensor_dict, **label_tensor_dict).items():
            if self.crop_size is not None:
                tensor = apply_indexing(tensor, dims, i, j, h, w)
            
            if key in input_tensor_dict:
                if "noise" in aug: #only add noise to input data
                    noise = torch.randn_like(tensor) * 0.01 
                    tensor = tensor + noise
                if "gaussianblur" in aug: #only blur input data
                    tensor = v2.GaussianBlur(kernel_size=5, sigma=1.5)(tensor)
                if self.channeldrop_prob !=0 and tensor.shape[-1] != 1: #only drop channels in input data
                    shapes = [1]*tensor.ndim
                    shapes[-1] = tensor.shape[-1]
                    mask = torch.rand(*shapes, device=tensor.device) > self.channeldrop_prob
                    tensor = tensor * mask
            
            if "verticalflip" in aug:
                tensor = torch.flip(tensor, dims=[dims[0]])  # vertical flip
            if "horizontalflip" in aug:
                tensor = torch.flip(tensor, dims=[dims[1]])  # horizontal flip
            if "rotate" in aug:
                tensor = torch.rot90(tensor, k=n_rotations, dims=dims)
            return_dic[key] = tensor

        return return_dic
    

def get_numerical_transform(data_cube, variable, scaler_mode, mask_value=0):
    if scaler_mode is None or scaler_mode == "" or scaler_mode.lower() == "none":
        return v2.Compose([v2.ToDtype(torch.float32),  MaskFillValue(mask_value=mask_value)])
    else:
        return v2.Compose([v2.ToDtype(torch.float32), CustomScaler(scaler_mode, data_cube[f"{variable}_norm"]), MaskFillValue(mask_value=mask_value)])
    
def get_categorical_transform(data_cube, variable):
    categories = data_cube[variable].attrs["flag_values"]
    return v2.Compose([v2.ToDtype(torch.long), CatToOneHot(categories)])
