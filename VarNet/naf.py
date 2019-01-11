from VarNet.torchkit.torchkit import flows, nn as nn_
from torch import nn


class NAF(nn.Module):
    def __init__(self, style_token_dim, z_dim):
        super(NAF, self).__init__()
        # Normalizing flow z* -> z
        self.style_token_dim = style_token_dim
        self.z_dim = z_dim
        self.context_dim = 450
        self.num_ds_layers = 1
        self.num_ds_dim = 16
        self.dim_h = 1920
        self.num_flow_layers = 1
        self.hidden_size = 1024
        self.embedding_dim = self.z_dim

        flowtype = 'ddsf'
        act = nn.ELU()
        if flowtype == 'affine':
            flow = flows.IAF
        elif flowtype == 'dsf':
            flow = lambda **kwargs: flows.IAF_DSF(num_ds_dim=self.num_ds_dim,
                                                  num_ds_layers=self.num_ds_layers,
                                                  **kwargs)
        elif flowtype == 'ddsf':
            flow = lambda **kwargs: flows.IAF_DDSF(num_ds_dim=self.num_ds_dim,
                                                   num_ds_layers=self.num_ds_layers,
                                                   **kwargs)
        self.context = nn.Sequential(
            nn.Linear(
                in_features=self.style_token_dim,
                out_features=self.context_dim
            ),
            act
        )

        self.inf = nn.Sequential(
            flows.LinearFlow(self.z_dim, self.context_dim),
            *[nn_.SequentialFlow(
                flow(dim=self.z_dim,
                     hid_dim=self.dim_h,
                     context_dim=self.context_dim,
                     num_layers=2,
                     activation=act),
                flows.FlipFlow(1)) for i in range(self.num_flow_layers)])

    def forward(self, z_star, logdet, style_vector):
        # compute context
        context = self.context(style_vector)
        z, logdet, context = self.inf((z_star, logdet, context))
        return z, logdet, context, z_star
