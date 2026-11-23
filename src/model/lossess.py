import torch
import torch.nn as nn

class LossBase(nn.Module):
    def __init__(self, no_data_value):
        super(LossBase, self).__init__()
        self.no_data_value = no_data_value

    def mask_no_data(self, target: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor):
        if len(target.shape) == len(pred.shape)-1:
            mask_no_data = target == self.no_data_value
            target = target.clone()
            target[mask_no_data] = 0 #just as a stackholder value for calculating the loss
        else:
            mask_no_data = torch.zeros_like(mask, dtype=torch.bool) #does not mask anything
        return target, (mask & ~mask_no_data).float()    
    

class SpatialMaskedCrossEntropyLoss(LossBase):
    def __init__(self, no_data_value, weight=None, entropy_weighted: bool=False, **loss_args):
        super(SpatialMaskedCrossEntropyLoss, self).__init__(no_data_value)
        self.criteria = torch.nn.CrossEntropyLoss(reduction='none', weight=weight, **loss_args)
        self.entropy_weighted = entropy_weighted

    def forward(self, predict: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, return_spatial=False, spatial_weights=1):
        target, mask = self.mask_no_data(target, predict, mask)
        
        disturb_loss_spatial = self.criteria(predict, target) # (n_batch, H, W)
        if return_spatial:
            return disturb_loss_spatial
        
        if self.entropy_weighted:
            probs = nn.functional.softmax(predict, dim=1).detach()
            entropy =  -(probs * torch.log(probs + 1e-8)).sum(dim=1)
            mask *= entropy  #relative entropy (normalized)

        if torch.is_tensor(spatial_weights):
            spatial_weights = spatial_weights.to(predict.device)
        #overall loss, not by each image (Since the data inside is a bit random)
        return (disturb_loss_spatial * mask * spatial_weights).sum() / (mask.sum().clamp_min(1.0))
        

class MaskedDiceLoss(LossBase):
    def __init__(self, no_data_value, weight=None, target_class = -1):
        super(MaskedDiceLoss, self).__init__(no_data_value)
        self.weight = weight
        self.target_class = target_class
        if self.weight is not None and target_class == -1:
            self.weight = weight / torch.sum(weight) # Normalized weight

    def forward(self, predict: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, spatial_weights= 1):
        N, C = predict.size()[:2]
        target, valid_mask = self.mask_no_data(target, predict, mask)
        predict = predict.reshape(N, C, -1) # (N, C, *)
        target = target.reshape(N, 1, -1).long() # (N, 1, *)
        valid_mask = valid_mask.reshape(N, 1, -1) # (N, 1, *)
        
        predict = nn.functional.softmax(predict, dim=1)  # (N, C, *)        
        ## convert target(N, 1, *) into one hot vector (N, C, *)
        target_onehot = torch.zeros_like(predict, dtype=torch.long) # (N, 1, *) ==> (N, C, *)
        target_onehot.scatter_(1, target, 1)   # (N, C, *)

        # Spatial weights: (N, 1, H*W)
        if not torch.is_tensor(spatial_weights):
            spatial_weights = torch.ones_like(valid_mask, dtype=predict.dtype)
        else:
            spatial_weights = spatial_weights.reshape(N, 1, -1).to(predict.device)
        weights = valid_mask * spatial_weights
        
        intersection = torch.sum(predict * target_onehot * weights, dim=2)  # (N, C)
        union = torch.sum(predict.pow(2) * weights, dim=2) + torch.sum(target_onehot * weights, dim=2)  # (N, C) 
        dice_coef = (2 * intersection) / (union.clamp_min(1.0))  # (N, C)

        if self.target_class != -1:
            dice_coef = dice_coef[:, self.target_class]  # (N,)
        elif self.weight is not None:
            dice_coef *= self.weight.to(predict.device) # (N, C)
        return 1 - torch.mean(dice_coef)  
    

class MaskedFocalLoss(LossBase):
    def __init__(self, no_data_value, alpha=1.0, gamma = 2.0, target_class = -1):
        super(MaskedFocalLoss, self).__init__(no_data_value)
        self.alpha = alpha
        self.target_class = target_class
        self.gamma = gamma
        
    def forward(self, predict: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        target, valid_mask = self.mask_no_data(target, predict, mask)

        log_probs = nn.functional.log_softmax(predict, dim=1)  # (B, C, H, W)
        probs = log_probs.exp()

        # Gather log-probabilities of the target class
        target = target.unsqueeze(1)  # (B, 1, H, W)
        log_pt = log_probs.gather(1, target).squeeze(1)  # (B, H, W)
        pt = probs.gather(1, target).squeeze(1)          # (B, H, W)

        focal_weight = self.alpha * (1.0 - pt) ** self.gamma
        loss = -focal_weight * log_pt

        if self.target_class != -1:
            class_mask = (target.squeeze(1) == self.target_class).float()
            valid_mask = valid_mask * class_mask

        # Apply mask and sum over all dimensions
        return (loss * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)