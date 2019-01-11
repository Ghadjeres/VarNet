import itertools

from torch import nn
import torch
import numpy as np
from torch.nn import Parameter

from VarNet.helpers import cuda_variable


class Feature(nn.Module):

    def __init__(self, attention_module=None):
        super(Feature, self).__init__()
        self.alpha_dim = None
        self.style_token_dim = None
        self.attention_module = attention_module

    def forward(self, x, m):
        """

        :param x:
        :param m:
        :return: style vector computed from x and m
        """

        alpha = self.alpha(x, m)
        style_vector = self.style_vector_from_alpha(alpha)
        return style_vector

    def alpha(self, x, m):
        raise NotImplementedError

    def alpha_iterator(self, max_num_dimensions=None, num_elements_per_dim=10):
        raise NotImplementedError


class SigmoidFeature(Feature):

    def __init__(self,
                 num_style_tokens,
                 style_token_dim,
                 attention_module,
                 aggregation_mode='mean'
                 ):
        """

        :param num_style_tokens:
        :param style_token_dim:
        :param attention_module:
        :param aggregation_mode: 'mean' or 'concat'
        """
        super(SigmoidFeature, self).__init__(attention_module=attention_module)
        self.aggregation_mode = aggregation_mode
        self.alpha_dim = num_style_tokens
        self.num_style_tokens = num_style_tokens
        self.style_token_dim = style_token_dim

        if self.aggregation_mode == 'mean':
            self.style_vector_dim = style_token_dim
        elif self.aggregation_mode == 'concat':
            self.style_vector_dim = style_token_dim * num_style_tokens
        else:
            raise NotImplementedError

        self.style_tokens = nn.Parameter(
            torch.randn(self.num_style_tokens, self.style_token_dim)
        )

    def random_alpha(self, batch_size):
        return cuda_variable(torch.rand(batch_size, self.num_style_tokens))

    def random_style_vector(self, batch_size, style_vector=None):
        """
        Alphas are drawn uniformly in [0, 1]
        :param batch_size:
        :param style_vector:
        :return:
        """
        random_alpha = self.random_alpha(batch_size)
        style_vector = self.style_vector_from_alpha(random_alpha)
        return style_vector

    def style_vector_from_alpha(self, alpha):
        # add batch dimension if alpha does not contain one
        if len(alpha.size()) == 1:
            alpha = alpha.unsqueeze(0)
        assert alpha.size(1) == self.alpha_dim
        batch_size = alpha.size(0)
        style_vector = (alpha.unsqueeze(2) *
                        self.style_tokens.unsqueeze(0).repeat(batch_size, 1, 1))
        if self.aggregation_mode == 'mean':
            style_vector = style_vector.mean(1)
        elif self.aggregation_mode == 'concat':
            style_vector = torch.cat(
                [t[:, 0, :]
                 for t in style_vector.split(split_size_or_sections=1,
                                             dim=1)],
                1)
        else:
            raise NotImplementedError
        return style_vector

    def alpha_iterator(self,
                       max_num_dimensions=None,
                       num_elements_per_dim=10,
                       offset=0):
        """

        :param max_num_dimensions:
        :param num_elements_per_dim:
        :param offset:
        :return: alphas have a batch_size of 1
        """
        if max_num_dimensions is None:
            max_num_dimensions = self.alpha_dim
        else:
            max_num_dimensions = min(self.alpha_dim, max_num_dimensions)
        g = itertools.product(np.arange(0., 1., 1 / num_elements_per_dim),
                              repeat=max_num_dimensions)
        remaining_dimensions = self.alpha_dim - max_num_dimensions

        begin = (0.5,) * offset
        end = (0.5,) * (remaining_dimensions - offset)

        alpha_gen = (cuda_variable(torch.Tensor(begin + t + end)).unsqueeze(0)
                     for t in g)
        return alpha_gen

    def alpha(self, x, m):
        return torch.sigmoid(self.attention_module(x, m))


class SimplexFeature(Feature):

    def __init__(self,
                 num_style_tokens,
                 style_token_dim,
                 attention_module,
                 ):
        """

        :param num_style_tokens: simplex dim
        :param style_token_dim:
        :param attention_module:
        """
        super(SimplexFeature, self).__init__(attention_module=attention_module)
        self.alpha_dim = num_style_tokens + 1
        self.num_style_tokens = num_style_tokens

        self.style_token_dim = style_token_dim
        self.style_vector_dim = style_token_dim

        self.style_tokens = nn.Parameter(
            torch.randn(self.num_style_tokens + 1, self.style_token_dim)
        )

    def random_alpha(self, batch_size):
        probs = torch.rand(batch_size, self.num_style_tokens + 1)
        probs = probs / probs.sum(1, keepdim=True)
        return cuda_variable(probs)

    def random_style_vector(self, batch_size, style_vector=None):
        """
        Alphas are drawn uniformly in [0, 1]
        :param batch_size:
        :param style_vector:
        :return:
        """
        random_alpha = self.random_alpha(batch_size)
        style_vector = self.style_vector_from_alpha(random_alpha)
        return style_vector

    def style_vector_from_alpha(self, alpha):
        # add batch dimension if alpha does not contain one
        if len(alpha.size()) == 1:
            alpha = alpha.unsqueeze(0)

        batch_size = alpha.size(0)
        style_vector = (alpha.unsqueeze(2) *
                        self.style_tokens.unsqueeze(0).repeat(batch_size, 1, 1))
        style_vector = style_vector.sum(1)
        return style_vector

    def alpha_iterator(self,
                       max_num_dimensions=None,
                       num_elements_per_dim=10,
                       offset=0):
        """

        :param max_num_dimensions:
        :param num_elements_per_dim:
        :param offset:
        :return: alphas have a batch_size of 1
        """
        if max_num_dimensions is None:
            max_num_dimensions = self.num_style_tokens + 1
        g = itertools.product(np.arange(0., 1., 1 / num_elements_per_dim),
                              repeat=max_num_dimensions)
        remaining_dimensions = self.num_style_tokens + 1 - max_num_dimensions

        begin = (0.01,) * offset
        end = (0.01,) * (remaining_dimensions - offset)

        alpha_gen = (cuda_variable(torch.Tensor(begin + t + end)).unsqueeze(0)
                     for t in g)
        probs_gen = (t / t.sum(1, keepdim=True)
                     for t in alpha_gen)
        return probs_gen

    def alpha(self, x, m):
        return torch.softmax(self.attention_module(x, m), 1)


class DiscreteFeature(Feature):

    def __init__(self,
                 num_values,
                 style_token_dim,
                 attention_module):
        """

        :param num_values:
        :param style_token_dim:
        :param attention_module: computes labels from x,
        the values of alpha_from_input are directly the embedding labels
        """
        super(DiscreteFeature, self).__init__(attention_module=attention_module)
        self.alpha_dim = style_token_dim
        self.num_values = num_values
        self.style_token_dim = style_token_dim
        self.style_tokens = nn.Embedding(num_embeddings=self.num_values,
                                         embedding_dim=self.style_token_dim)

    def random_alpha(self, batch_size):
        r_a = torch.randint(low=0,
                            high=self.num_values,
                            size=(batch_size,)
                            ).long()

        return cuda_variable(r_a)

    def random_style_vector(self, batch_size, style_vector=None):
        """
        Randomize the style_vector tensor in order to sample
        style vectors with their probability in the dataset
        batch_size argument is ignored
        :param batch_size:
        :param style_vector:
        :return:
        """

        random_style_vector = style_vector[
            torch.randperm(style_vector.size(0))
        ]
        return random_style_vector

    def style_vector_from_alpha(self, alpha):
        return self.style_tokens(alpha)

    def alpha_iterator(self, max_num_dimensions=None, num_elements_per_dim=None):
        # assert max_num_dimensions is None or max_num_dimensions == 1
        assert num_elements_per_dim is None or max_num_dimensions <= self.num_values
        return (
            cuda_variable(torch.Tensor([d]).long())
            for d in range(self.num_values)
        )

    def alpha(self, x, m):
        return self.attention_module(x, m)


class CombinedFeature(Feature):
    def __init__(self, list_of_features):
        super(CombinedFeature, self).__init__()

        self.list_of_features = nn.ModuleList(
            list_of_features
        )

        self.alpha_dim = sum([
            feature.alpha_dim
            for feature in self.list_of_features])

        self.style_token_dim = sum([
            feature.style_token_dim
            for feature in self.list_of_features]
        )

    def forward(self, x, m):
        style_vector_list = [
            features.forward(x, m)
            for features in self.list_of_features
        ]
        style_vector = torch.cat(style_vector_list, 1)
        return style_vector

    def style_vector_from_alpha(self, alpha):
        """

        :param alpha: list of alphas
        :return:
        """
        # split_sizes = [feature.alpha_dim for feature in self.list_of_features]
        # alpha_list = torch.split(alpha,
        #                          split_size_or_sections=split_sizes,
        #                          dim=1)
        alpha_list = alpha
        assert len(alpha_list) == len(self.list_of_features)

        style_vectors = [
            feature.style_vector_from_alpha(alpha)
            for feature, alpha in zip(self.list_of_features, alpha_list)
        ]
        return torch.cat(style_vectors, 1)

    def random_style_vector(self, batch_size, style_vector=None):
        """
        Randomize the style_vector tensor in order to sample
        style vectors with their probability in the dataset
        batch_size argument is ignored
        :param batch_size:
        :param style_vector:
        :return:
        """
        split_sizes = [feature.style_token_dim for feature in self.list_of_features]

        style_vector_list = torch.split(style_vector,
                                        split_size_or_sections=split_sizes,
                                        dim=1)
        random_style_vectors = [
            feature.random_style_vector(batch_size, style_vector=style_vector)
            for feature, style_vector in zip(self.list_of_features, style_vector_list)
        ]
        return torch.cat(random_style_vectors, 1)

    def alpha_iterator(self, max_num_dimensions=None, num_elements_per_dim=None):
        # todo how to handle max_num_dimensions?
        # Cannot use that with discrete and continuous alphas...
        iterator_list = [
            feature.alpha_iterator(max_num_dimensions=1,
                                   num_elements_per_dim=num_elements_per_dim)
            for feature in self.list_of_features
        ]

        # iterator_list = [
        #     self.list_of_features[0].alpha_iterator(max_num_dimensions=2,
        #                                             num_elements_per_dim=num_elements_per_dim),
        #     self.list_of_features[1].alpha_iterator(max_num_dimensions=0,
        #                                             num_elements_per_dim=1)
        # ]
        # iterator_list = [
        #     self.list_of_features[0].alpha_iterator(max_num_dimensions=1,
        #                                             num_elements_per_dim=1),
        #     self.list_of_features[1].alpha_iterator(max_num_dimensions=2,
        #                                             num_elements_per_dim=num_elements_per_dim)
        # ]

        # labels = self.list_of_features[1].alpha_iterator()
        # next(labels)
        # one_label = next(labels)
        # iterator_list = [
        #     self.list_of_features[0].alpha_iterator(), (one_label for _ in range(1))
        # ]
        cartesian_prod = itertools.product(*iterator_list)

        # return (
        #     torch.cat(t, 0)
        #     for t in cartesian_prod
        # )
        return cartesian_prod

    def random_alpha(self, batch_size):
        alpha_list = [
            feature.random_alpha(batch_size)
            for feature in self.list_of_features
        ]
        return alpha_list

    def alpha(self, x, m):
        alpha_list = [
            feature.alpha(x, m)
            for feature in self.list_of_features
        ]
        return alpha_list


class HierarchicalFeature(Feature):
    """
    Combines Sigmoid and Discrete Feature
    alpha = label, alpha_sigmoid
    """

    def __init__(self,
                 num_values,
                 num_style_tokens_per_label,
                 style_token_dim,
                 attention_module):
        super(HierarchicalFeature, self).__init__(attention_module=attention_module)
        # todo aggregation_mode is 'mean'
        self.alpha_dim = style_token_dim
        self.num_values = num_values
        self.num_style_tokens_per_label = num_style_tokens_per_label
        self.style_token_dim = style_token_dim
        self.style_tokens = nn.Embedding(
            num_embeddings=num_values,
            embedding_dim=self.style_token_dim * self.num_style_tokens_per_label)
        self.style_tokens_bias = nn.Embedding(
            num_embeddings=num_values,
            embedding_dim=self.style_token_dim)

    def random_alpha(self, batch_size):
        rand_label = cuda_variable(torch.randint(low=0,
                                                 high=self.num_values,
                                                 size=(batch_size,)
                                                 ).long())
        rand_alpha_sigmoid = cuda_variable(torch.rand(batch_size,
                                                      self.num_style_tokens_per_label))
        return rand_label, rand_alpha_sigmoid

    def style_vector_from_alpha(self, alpha):
        label, alpha_sigmoid = alpha
        batch_size = label.size(0)

        style_tokens = self.style_tokens(label)
        style_tokens = style_tokens.view(batch_size,
                                         self.num_style_tokens_per_label,
                                         self.style_token_dim)
        style_vector = alpha_sigmoid.unsqueeze(2) * style_tokens
        style_vector = style_vector.sum(1)

        style_vector_bias = self.style_tokens_bias(label)
        # return style_vector
        return style_vector + style_vector_bias

    def alpha_iterator(self, max_num_dimensions=None,
                       num_elements_per_dim=10,
                       offset=0):
        """

        :param max_num_dimensions: Only for sigmoid features
        :param num_elements_per_dim:
        :param offset:
        :return:
        """
        label_iterator = (
            cuda_variable(torch.Tensor([d]).long())
            for d in range(self.num_values)
        )

        if max_num_dimensions is None:
            max_num_dimensions = self.num_style_tokens_per_label
        else:
            # label is always the first dimension
            max_num_dimensions = max_num_dimensions - 1
            # todo to remove, investigate why this does not break
            # max_num_dimensions = max_num_dimensions
        g = itertools.product(np.arange(0., 1., 1 / num_elements_per_dim),
                              repeat=max_num_dimensions)
        remaining_dimensions = self.num_style_tokens_per_label - max_num_dimensions

        begin = (0.5,) * offset
        end = (0.5,) * (remaining_dimensions - offset)

        alpha_sigmoid_iterator = (
            cuda_variable(torch.Tensor(begin + t + end)).unsqueeze(0)
            for t in g)

        return itertools.product(label_iterator, alpha_sigmoid_iterator)

    def random_style_vector(self, batch_size, style_vector=None):
        """
        Alphas are drawn uniformly in [0, 1]
        :param batch_size:
        :param style_vector:
        :return:
        """
        random_alpha = self.random_alpha(batch_size)
        style_vector = self.style_vector_from_alpha(random_alpha)
        return style_vector

    def alpha(self, x, m):
        return m, torch.sigmoid(self.attention_module(x, m))


class EmptyFeature(Feature):
    def __init__(self,
                 num_style_tokens=1,
                 style_token_dim=1,
                 attention_module=None,
                 aggregation_mode='mean'
                 ):
        """

        :param num_style_tokens:
        :param style_token_dim:
        :param attention_module:
        :param aggregation_mode: 'mean' or 'concat'
        """
        super(EmptyFeature, self).__init__()
        self.alpha_dim = num_style_tokens
        self.num_style_tokens = num_style_tokens
        self.style_token_dim = style_token_dim
        self.empty_params = Parameter(torch.zeros(1,))

    def random_alpha(self, batch_size):
        return cuda_variable(torch.zeros(batch_size, self.num_style_tokens))

    def random_style_vector(self, batch_size, style_vector=None):
        """
        Alphas are drawn uniformly in [0, 1]
        :param batch_size:
        :param style_vector:
        :return:
        """
        random_alpha = self.random_alpha(batch_size)
        style_vector = self.style_vector_from_alpha(random_alpha)
        return style_vector

    def style_vector_from_alpha(self, alpha):
        return cuda_variable(torch.zeros((alpha.size(0), self.style_token_dim)))

    def alpha_iterator(self,
                       max_num_dimensions=None,
                       num_elements_per_dim=10,
                       offset=0):
        alpha_gen = (cuda_variable(torch.zeros(1))
                     for _ in range(1))
        return alpha_gen

    def alpha(self, x, m):
        # TODO only used for debug
        if isinstance(x, tuple) or isinstance(x, list):
            batch_size = x[0].size(0)
        else:
            batch_size = x.size(0)
        return cuda_variable(torch.zeros(batch_size))
