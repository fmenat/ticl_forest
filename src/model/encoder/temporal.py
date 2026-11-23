import torch

from ..layers.simple import PositionalEncoder,EmbeddingLayer

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

    def get_channels_out(self):
        return self.latent_dims if self.latent_dims > 0 else self.input_size
    
class RNNet(Base_Encoder):
    def __init__(
        self,
        n_feats: int,
        start_filters: int = 128,
        n_layers: int = 1,
        dropout: float =0,
        bidirectional: bool = False,
        unit_type: str="gru",
        emb_feats: int = -1,
        **kwargs,
    ):
        super().__init__(input_size=start_filters, latent_dims=emb_feats, **kwargs)
        self.unit_type = unit_type.lower()
        self.n_feats = n_feats
        self.start_filters = start_filters
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        if self.unit_type == "gru":
            rnn_type_class = torch.nn.GRU
        elif self.unit_type == "lstm":
            rnn_type_class = torch.nn.LSTM
        elif self.unit_type == "rnn":
            rnn_type_class = torch.nn.RNN
        else:
            raise Exception("unit_type not recognized, available options are [rnn, gru, lstm]")

        self.rnn = rnn_type_class(
                input_size=self.n_feats,
                hidden_size=self.start_filters,
                num_layers=self.n_layers,
                batch_first=True,
                dropout=self.dropout,
                bidirectional=self.bidirectional)
    
    def forward(self, x: torch.Tensor, **kwargs):
        rnn_out, states = self.rnn(x)
        states = states[0] if type(states) == tuple else states
        if self.bidirectional:
            rnn_out = rnn_out.view(-1, rnn_out.shape[1], 2, self.start_filters)
            rnn_out = (rnn_out[:,:, 0,: ] + rnn_out[:,:, 1,: ])/2

            if self.output_state:
                states = states.view(2, self.n_layers, -1, self.start_filters)
                states = (states[0] + states[1])/2

        if self.latent_dims > 0:
            out_ = self.embedding_layer(rnn_out[:, -1]) # get last time ste
        else:
            out_ = rnn_out[:, -1]
        return {"repre": out_}


class TransformerNet(Base_Encoder):
    def __init__(
        self,
        n_feats: int,
        start_filters: int = 128,
        dropout: float = 0.0,
        n_layers: int = 1,
        num_heads: int = 1,
        len_max_seq: int = 24,
        fixed_pos_encoding: bool = True,
        activation = "relu",
        dim_feedforward = 2048,
        emb_feats: int = -1,
        **kwargs,
    ):
        super().__init__(input_size=start_filters, latent_dims=emb_feats, **kwargs)
        self.n_feats = n_feats
        self.start_filters = start_filters
        self.n_layers = n_layers
        self.dropout = dropout
        self.len_max_seq = len_max_seq+ 1 #for class token
        self.num_heads = num_heads
        self.fixed_pos_encoding = fixed_pos_encoding

        self.token_layer = EmbeddingLayer(in_features=self.n_feats, out_features=self.start_filters, model_type="linear")

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, self.emb_dim))  
        self.pos_encoder = PositionalEncoder(
            d_model=self.emb_dim, 
            max_len=self.len_max_seq)
        
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.emb_dim,
            nhead=self.num_heads,
            dropout=self.dropout,
            dim_feedforward=dim_feedforward,
            activation=activation,
            batch_first=True,
        )
        self.tr_encoder = torch.nn.TransformerEncoder(encoder_layer=encoder_layer,num_layers=self.n_layers)

    def forward(self, x, src_key_padding_mask=None, **kwargs):
        x = self.token_layer(x)
        # add cls token to input sequence
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), 1)

        # add time step for cls token in src_key_padding_mask
        if src_key_padding_mask is not None:
            src_key_padding_mask = torch.cat(
                (
                    torch.zeros_like(src_key_padding_mask[:, -1].reshape(-1, 1)).bool(),  #always false to attend cls token
                    src_key_padding_mask,  # for rest fo x
                ),
                dim=1)
        if self.fixed_pos_encoding: # add position encoding
            x = self.pos_encoder(x)
        
        # pass x through transformer encoder
        x = self.tr_encoder(x, src_key_padding_mask=src_key_padding_mask)

        if self.latent_dims > 0:
            x = self.embedding_layer(x[:, 0, :])
        else:
            x = x[:, 0, :]# extract class token feature: 
        
        return {"repre": x} 