def reset_weights(model):
    for m in model.modules():
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()