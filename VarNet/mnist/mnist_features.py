from torch import nn

from VarNet.features import SigmoidFeature, DiscreteFeature, HierarchicalFeature
from VarNet.mnist.fashion_mnist_encoder import FashionMNISTEncoder
from VarNet.mnist.mnist_encoder import MNISTEncoder, MNISTLabeledEncoder, \
    MNISTLabeledIndependantEncoder, MNISTLabeledSemiIndependantEncoder


class MNISTSigmoidFeature(SigmoidFeature):
    def __init__(self,
                 num_style_tokens,
                 style_token_dim,
                 attention_hidden_size=400):
        attention_module = MNISTEncoder(
            num_style_tokens=num_style_tokens,
            hidden_size=attention_hidden_size,
            activation=lambda x: x
        )
        super(MNISTSigmoidFeature, self).__init__(
            num_style_tokens=num_style_tokens,
            style_token_dim=style_token_dim,
            attention_module=attention_module
        )


class FashionMNISTSigmoidFeature(SigmoidFeature):
    def __init__(self,
                 num_style_tokens,
                 style_token_dim,
                 attention_hidden_size=400):
        attention_module = FashionMNISTEncoder(
            num_style_tokens=num_style_tokens,
            hidden_size=attention_hidden_size,
            activation=lambda x: x
        )
        super(FashionMNISTSigmoidFeature, self).__init__(
            num_style_tokens=num_style_tokens,
            style_token_dim=style_token_dim,
            attention_module=attention_module
        )


class MNISTLabelFeature(DiscreteFeature):
    # todo chooose that alpha = attention(x, m)
    def __init__(self, style_token_dim):
        attention_module = lambda x, label: label
        super(MNISTLabelFeature, self).__init__(
            num_values=10,
            style_token_dim=style_token_dim,
            attention_module=attention_module
        )


class MNISTLocalSigmoidFeature(HierarchicalFeature):
    def __init__(self,
                 num_values,
                 num_style_tokens_per_label,
                 style_token_dim,
                 attention_hidden_size=400
                 ):
        attention_module = MNISTLabeledEncoder(
            num_style_tokens=num_style_tokens_per_label,
            label_embedding_dim=16,
            activation=lambda x: x)
        super(MNISTLocalSigmoidFeature, self).__init__(
            num_values=num_values,
            num_style_tokens_per_label=num_style_tokens_per_label,
            style_token_dim=style_token_dim,
            attention_module=attention_module)


class MNISTLocalIndependantSigmoidFeature(HierarchicalFeature):
    def __init__(self,
                 num_values,
                 num_style_tokens_per_label,
                 style_token_dim,
                 attention_hidden_size=400
                 ):
        attention_module = MNISTLabeledIndependantEncoder(
            num_style_tokens=num_style_tokens_per_label,
            label_embedding_dim=16,
            activation=lambda x: x
        )
        super(MNISTLocalIndependantSigmoidFeature, self).__init__(
            num_values=num_values,
            num_style_tokens_per_label=num_style_tokens_per_label,
            style_token_dim=style_token_dim,
            attention_module=attention_module)


class MNISTLocalSemiIndependantSigmoidFeature(HierarchicalFeature):
    def __init__(self,
                 num_values,
                 num_style_tokens_per_label,
                 style_token_dim,
                 attention_hidden_size=400
                 ):
        attention_module = MNISTLabeledSemiIndependantEncoder(
            num_style_tokens=num_style_tokens_per_label,
            label_embedding_dim=16,
            activation=lambda x: x
        )
        super(MNISTLocalSemiIndependantSigmoidFeature, self).__init__(
            num_values=num_values,
            num_style_tokens_per_label=num_style_tokens_per_label,
            style_token_dim=style_token_dim,
            attention_module=attention_module)
