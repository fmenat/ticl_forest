import torch

class Base_Encoder(torch.nn.Module):
    """
        it adds a linear layer at the end of an encoder model with possible batch normalization.
    """
    def __init__(
        self,
        input_size: int,
        latent_dims: int,
        norm_last: bool = False, #it is assumed it is layer norm
        bnorm_last: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.norm_last = norm_last
        self.bnorm_last = bnorm_last
        self.input_size = input_size
        self.latent_dims = latent_dims

        if self.latent_dims > 0:
            layer_list = [torch.nn.Linear(self.input_size, self.latent_dims)]
            if self.norm_last:
                layer_list.append(torch.nn.LayerNorm(self.latent_dims))
            elif self.bnorm_last:
                layer_list.append(torch.nn.BatchNorm1d(self.latent_dims))
            
            self.embedding_layer = torch.nn.Sequential(*layer_list)

    def get_output_size(self):
        return self.latent_dims if self.latent_dims > 0 else self.input_size
    
class MLP(Base_Encoder):
    def __init__(
        self,
        n_feats: int,
        start_filters: int = 128,
        n_layers: int = 1,
        dropout: float =0,
        emb_feats: int = -1,
        act: str ="leakyrelu",
        **kwargs,
    ):
        super().__init__(input_size=n_feats, latent_dims=emb_feats, **kwargs)
        
        layer_sizes = [n_feats] + [start_filters]*n_layers
        layers = []
        for l_id in range(len(layer_sizes) - 1):
            layers.append(
                torch.nn.Sequential(
                    torch.nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]),
                    torch.nn.BatchNorm1d(layer_sizes[l_id+1]), 
                    torch.nn.LeakyReLU(inplace=True) if act == "leakyrelu" else torch.nn.GELU(approximate='tanh'),
                    torch.nn.Dropout(p=dropout) if dropout!=0 else torch.nn.Identity(),
                )
            )
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x, **kwargs):
        out =  self.layers(x)
        if self.latent_dims > 0:
            out = self.embedding_layer(out)
        return {"repre": out}