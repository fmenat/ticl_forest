
#define a class that takes an original model and add a copy of a specific module/layer when required

import torch.nn as nn
import copy 

def set_eval_mode(model, avoid="step_modules"):
    for name, module in model.named_children():
        if any(module is sm for sm in get_submodule(model, avoid)):
            continue
        module.eval()   # freezes BN running‑mean/var, disables dropout

def freeze_backbone(model):
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze only step_modules
    for param in model.step_modules.parameters():
        param.requires_grad = True    

def freeze_and_set_eval_backbone(model, freeze_flag, eval_flag, avoid="step_modules"):
    if freeze_flag:
        freeze_backbone(model)
    if eval_flag:
        set_eval_mode(model, avoid=avoid)


def update_modular_incremental_model(base_model, module_factory: list):
    base_model.module_factory = module_factory

    base_model.step_modules = nn.ModuleDict()
    base_model.active_step = None

    #create and store store original modules once
    base_model._original_modules = {
        name: get_submodule(base_model, name)
        for name in base_model.module_factory
    }
    return base_model

def add_step_module(base_model, step_name):
    #add new layer for a given step (ONLY call when changing task)
    base_model.step_modules[step_name] = nn.ModuleDict()

    for name in base_model.module_factory:
        old_module = base_model._original_modules[name]
        step_module = copy.deepcopy(old_module)
        step_module.requires_grad_(True)

        safe_name = name.replace(".", "__")
        base_model.step_modules[step_name][safe_name] = step_module

def activate_step(base_model, step_name):
    # Switch base_model to a given step (ONLY call when changing task)
    assert step_name in base_model.step_modules, f"{step_name} not found"
    base_model.active_step = step_name

    for name in base_model.module_factory:
        safe_name = name.replace(".", "__")
        step_module = base_model.step_modules[step_name][safe_name]

        set_submodule(base_model, name, step_module)


#### UTILS
def get_submodule(model, path):
    parts = path.split(".")
    mod = model

    for p in parts:
        if isinstance(mod, nn.ModuleList):
            mod = mod[int(p)]
        elif isinstance(mod, nn.ModuleDict):
            mod = mod[p]
        else:
            mod = getattr(mod, p)

    return mod

def set_submodule(model, path, step_module):
    parts = path.split(".")
    parent = model

    for p in parts[:-1]:
        if isinstance(parent, nn.ModuleList):
            parent = parent[int(p)]
        elif isinstance(parent, nn.ModuleDict):
            parent = parent[p]
        else:
            parent = getattr(parent, p)

    last = parts[-1]
    if isinstance(parent, nn.ModuleList):
        parent[int(last)] = step_module
    elif isinstance(parent, nn.ModuleDict):
        parent[last] = step_module
    else:
        setattr(parent, last, step_module)