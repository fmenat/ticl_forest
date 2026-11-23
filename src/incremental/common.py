import copy


def set_old_loss(model):
    model.standard_loss = model.loss_batch


