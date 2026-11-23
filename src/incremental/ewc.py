import torch
import copy
from torch import nn

def set_old_loss(model):
    model.standard_loss = model.loss_batch

def get_model_params(model):
    return model.named_parameters()

def copy_fixed_params(model, device=None):
    return {n: p.clone().detach().to(device) for n, p in get_model_params(model) if p.requires_grad}


def compute_online_fisher(model, dataloader, old_fisher= None, gamma=0.95, device=None):
    was_training = model.training
    model.eval()
    original_device = model.device
    model.to(device)

    fisher = {n: torch.zeros_like(p) for n, p in get_model_params(model) if p.requires_grad}
    for batch in dataloader:
        model.zero_grad()
        loss_dict = model.standard_loss(batch, val_loss=True) 
        loss = loss_dict["ce"] #["loss"]
        loss.backward()

        for n, p in get_model_params(model):
            if p.grad is not None:
                fisher[n] += p.grad.detach()**2
    

    for n in fisher: #normalize by the number of samples in the dataloader
        fisher[n] /= len(dataloader)

    normalize_val = torch.cat( [f.flatten() for f in fisher.values()]).mean()
    for n in fisher: #normalize all fisher values to avoid large values
        fisher[n] /= normalize_val


    #combine with old fisher
    if old_fisher is not None:
        for n in fisher:
            fisher[n] = gamma * old_fisher[n].to(device) + fisher[n]
    
    #back model to original
    model.to(original_device)
    if was_training:
        model.train()
    return fisher


def ewc_modify_loss(model, old_params, fisher, lambda_ewc=1.0, **kwargs):
    def new_loss_batch(batch: dict):
        model_device = model.device
        loss_dict = model.standard_loss(batch)
        if not model.training:
            return loss_dict

        ewc_loss = 0.0
        for n, p in get_model_params(model):
            if n in fisher:
                ewc_loss += (fisher[n] * (p- old_params[n].to(model_device) )**2).sum()/2
        loss_dict["ewc_loss"] =  ewc_loss
        
        total_loss = loss_dict["loss"] + lambda_ewc*loss_dict["ewc_loss"]
        loss_dict["loss"] = total_loss

        return loss_dict
    
    model.loss_batch = new_loss_batch