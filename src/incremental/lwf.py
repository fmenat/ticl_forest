import copy
import torch
from torch import nn


def copy_fixed_model(old_model, device=None):
    copy_model = copy.deepcopy(old_model)
    #freeze old model parameters
    for p in copy_model.parameters():
        p.requires_grad = False
    copy_model.eval()
    copy_model.requires_grad_(False)
    return copy_model.to(device)

def lwf_modify_loss(IL_model, old_model, temperature=2, alpha=0.5):
    def new_loss_batch(batch: dict):
        #calculate the original loss
        loss_dic = IL_model.standard_loss(batch)
        if not IL_model.training:
            return loss_dic
        
        new_logprob_temp = nn.functional.log_softmax(loss_dic["pred_logits"] / temperature, dim=1)

        with torch.no_grad():
            if old_model.device != IL_model.device:
                old_model.to(IL_model.device)
            input_sensor, reference_labels = old_model.prepare_batch(batch)
            masks = old_model.mask_calculation(reference_labels)
            old_output = old_model(input_sensor)
        old_logprob_temp = nn.functional.log_softmax(old_output["logits"] / temperature, dim=1).detach()


        #calculate the distillation loss
        kl_map = nn.functional.kl_div(new_logprob_temp, old_logprob_temp, reduction='none', log_target=True) 
        kl_map = kl_map.sum(dim=1)   # → [batch, height, width]

        loss_dic["distillation_loss"] = (temperature**2)* (kl_map * masks["valid_mask"]).sum() / masks["valid_mask"].sum()
        
        #combine the losses
        total_loss = loss_dic["loss"] + alpha* loss_dic["distillation_loss"]
        loss_dic["loss"] = total_loss

        return loss_dic
    
    IL_model.loss_batch = new_loss_batch
