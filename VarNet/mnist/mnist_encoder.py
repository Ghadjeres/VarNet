from VarNet.torchkit.torchkit import flows
from VarNet.torchkit.torchkit import nn as nn_  # torchkit

import torch
from torch import nn
import torch.nn.functional as F

from VarNet.helpers import cuda_variable


class MNISTNAFEncoder(nn.Module):
    def __init__(self, z_dim, hidden_size=400):
        super(MNISTNAFEncoder, self).__init__()
        self.z_dim = z_dim
        self.context_dim = 450
        self.num_ds_layers = 1
        self.num_ds_dim = 16
        self.dim_h = 1920
        self.num_flow_layers = 1

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
            nn.Linear(784, hidden_size),
            act,
            nn.Linear(hidden_size, self.context_dim),
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

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        context = self.context(x)

        # normalizing flow
        noise = cuda_variable(torch.randn(batch_size, self.z_dim))
        logdet = cuda_variable(torch.zeros(batch_size))
        z_samples, logdet, _ = self.inf((noise, logdet, context))
        return z_samples, logdet, context, noise


class MNISTEncoder(nn.Module):
    def __init__(self, num_style_tokens,
                 hidden_size=400,
                 activation=nn.Sigmoid(),
                 dropout=None):
        super(MNISTEncoder, self).__init__()
        self.fc1 = nn.Linear(784, hidden_size)
        self.fc21 = nn.Linear(hidden_size,
                              num_style_tokens)
        self.activation = activation
        self.dropout = dropout

    def forward(self, x, m):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        h1 = torch.relu(self.fc1(x))
        if self.dropout is not None:
            h1 = F.dropout(h1, p=self.dropout, training=self.training)
        return self.activation(self.fc21(h1))


class MNISTLabeledEncoder(nn.Module):
    def __init__(self, num_style_tokens,
                 label_embedding_dim,
                 hidden_size=400,
                 activation=nn.Sigmoid(),
                 dropout=None):
        super(MNISTLabeledEncoder, self).__init__()
        self.label_embedding = nn.Embedding(num_embeddings=10,
                                            embedding_dim=label_embedding_dim)
        self.fc1 = nn.Linear(784 + label_embedding_dim, hidden_size)
        self.fc21 = nn.Linear(hidden_size,
                              num_style_tokens)
        self.activation = activation
        self.dropout = dropout

    def forward(self, x, m):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        label_embedded = self.label_embedding(m)

        h0 = torch.cat([x, label_embedded], 1)
        h1 = torch.relu(self.fc1(h0))
        if self.dropout is not None:
            h1 = F.dropout(h1, p=self.dropout, training=self.training)
        return self.activation(self.fc21(h1))


class MNISTLabeledIndependantEncoder(nn.Module):
    def __init__(self, num_style_tokens,
                 label_embedding_dim=400,
                 hidden_size=400,
                 activation=nn.Sigmoid(),
                 dropout=None):
        super(MNISTLabeledIndependantEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_style_tokens = num_style_tokens

        # todo better the other way
        self.fc1 = nn.Embedding(num_embeddings=10,
                                embedding_dim=(784 + 1) * self.hidden_size)
        self.fc2 = nn.Embedding(
            num_embeddings=10,
            embedding_dim=(self.hidden_size + 1) * self.num_style_tokens)

        # normalize
        self.fc1.weight.data = self.fc1.weight.data / ((784 + 1) * self.hidden_size)
        self.fc2.weight.data = (
                self.fc2.weight.data / ((self.hidden_size + 1) * self.num_style_tokens))

        # self.fc1 = nn.Sequential(
        #     nn.Embedding(num_embeddings=10,
        #                  embedding_dim=label_embedding_dim),
        #     nn.Linear(label_embedding_dim, (784 + 1) * self.hidden_size)
        # )
        # # renormalize
        # weights_fc1, bias_fc1 = list(self.fc1[1].parameters())
        # weights_fc1.data = weights_fc1.data / (785 * self.hidden_size)
        # bias_fc1.data = bias_fc1.data / self.hidden_size
        #
        # self.fc2 = nn.Sequential(
        #     nn.Embedding(
        #         num_embeddings=10,
        #         embedding_dim=label_embedding_dim),
        #     nn.Linear(label_embedding_dim, (self.hidden_size + 1) * self.num_style_tokens)
        # )
        #
        # # renormalize
        # weights_fc2, bias_fc2 = list(self.fc2[1].parameters())
        # weights_fc2.data = weights_fc2.data / ((self.hidden_size + 1) * self.num_style_tokens)
        # bias_fc2.data = bias_fc2.data / self.hidden_size

        self.activation = activation
        self.dropout = dropout

    def forward(self, x, m):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        fc1 = self.fc1(m).view(batch_size, (784 + 1), self.hidden_size)
        fc1_w = fc1[:, :784, :]
        fc1_bias = fc1[:, -1, :]
        fc2 = self.fc2(m).view(
            batch_size, (self.hidden_size + 1), self.num_style_tokens)
        fc2_w = fc2[:, :self.hidden_size, :]
        fc2_bias = fc2[:, -1, :]

        h1 = torch.einsum('bkj,bk->bj', (fc1_w, x)) + fc1_bias
        h1 = torch.relu(h1)

        if self.dropout is not None:
            h1 = F.dropout(h1, p=self.dropout, training=self.training)

        h2 = torch.einsum('bkj,bk->bj', (fc2_w, h1)) + fc2_bias
        return self.activation(h2)


class MNISTLabeledSemiIndependantEncoder(nn.Module):
    def __init__(self, num_style_tokens,
                 label_embedding_dim=400,
                 hidden_size=400,
                 activation=nn.Sigmoid(),
                 dropout=None):
        super(MNISTLabeledSemiIndependantEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.label_embedding = nn.Embedding(num_embeddings=10,
                                            embedding_dim=label_embedding_dim)
        self.label_to_weights = nn.Linear(label_embedding_dim, (784 + 1) * self.hidden_size)
        self.fc21 = nn.Linear(self.hidden_size,
                              num_style_tokens)
        self.activation = activation
        self.dropout = dropout

    def forward(self, x, m):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        label_embedded = self.label_embedding(m)
        fc1 = self.label_to_weights(label_embedded)
        fc1 = fc1.view(batch_size, (784 + 1), self.hidden_size)
        mat = fc1[:, :784, :]
        bias = fc1[:, -1, :]
        h1 = torch.einsum('bkj,bk->bj', (mat, x)) + bias
        h1 = torch.relu(h1)

        if self.dropout is not None:
            h1 = F.dropout(h1, p=self.dropout, training=self.training)
        return self.activation(self.fc21(h1))
