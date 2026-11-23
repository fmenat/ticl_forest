import torch.nn as nn
import copy

from .simple import ConvLSTMCell

class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dims, kernel_size=3):
        super().__init__()
        self.num_layers = len(hidden_dims)
        self.cells = nn.ModuleList()

        for i in range(self.num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dims[i - 1]
            self.cells.append(ConvLSTMCell(cur_input_dim, hidden_dims[i], kernel_size))

    def forward(self, x):
        """
        x: [B, T, C, H, W]
        Returns:
            final hidden state of last layer
        """
        b, t, _, h, w = x.size()
        device = x.device

        h_t = []
        c_t = []
        for cell in self.cells:   # Initialize hidden states for all layers
            h_i, c_i = cell.init_hidden(b, (h, w), device)
            h_t.append(h_i)
            c_t.append(c_i)

        for i in range(t): # Iterate over time steps
            input_ = x[:, i]
            for layer_idx, cell in enumerate(self.cells): # Iterate over layers
                h_next, c_next = cell(input_, (h_t[layer_idx], c_t[layer_idx]))
                h_t[layer_idx], c_t[layer_idx] = h_next, c_next
                input_ = h_next  # pass to next layer
        return h_t[-1]  # output from last layer
    

