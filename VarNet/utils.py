from torch import nn
import torch
from torch.distributions import kl_divergence, Normal
from PIL import Image
import numpy as np


def categorical_crossentropy(value, target):
    """

    :param value: list of (batch_size, chorale_length, num_notes)
    :param target: (batch_size, num_voices, chorale_length)
    :return:
    """
    # put num_voices first
    target = target.transpose(0, 1)

    cross_entropy = nn.CrossEntropyLoss(size_average=False, reduce=False)
    sum = 0
    for voice, voice_target in zip(value, target):
        # voice is (batch_size, chorale_length, num_notes)
        # voice_target is (batch_size, chorale_length)
        # put time first
        voice = voice.transpose(0, 1)
        voice_target = voice_target.transpose(0, 1)
        for note_probs, label in zip(voice, voice_target):
            ce = cross_entropy(note_probs, label)
            sum += ce
    return sum


def elbo(weights_per_voice, chorale, z_distribution, beta):
    prior = Normal(torch.zeros_like(z_distribution.loc),
                   torch.ones_like(z_distribution.scale)
                   )
    kl = kl_divergence(z_distribution, prior).sum(1)

    ce = categorical_crossentropy(weights_per_voice, chorale)
    # TODO check signs
    return -ce - beta * kl, ce, kl


def glsr_elbo(weights_per_voice, chorale, z_distribution, beta, log_var_0):
    scale = torch.ones_like(z_distribution.scale)
    scale[:, 0] = torch.exp(log_var_0)

    prior = Normal(torch.zeros_like(z_distribution.loc),
                   scale
                   )

    kl = kl_divergence(z_distribution, prior).sum(1)
    ce = categorical_crossentropy(weights_per_voice, chorale)
    return - ce - beta * kl, ce, kl


def compute_kernel(x, y, k):
    batch_size_x, dim_x = x.size()
    batch_size_y, dim_y = y.size()
    assert dim_x == dim_y

    xx = x.unsqueeze(1).expand(batch_size_x, batch_size_y, dim_x)
    yy = y.unsqueeze(0).expand(batch_size_x, batch_size_y, dim_y)
    distances = (xx - yy).pow(2).sum(2)
    return k(distances)


def mmd_loss(weights, chorale, z_tilde, z, coeff=10):
    # gaussian
    def gaussian(d, var=16.):
        return torch.exp(- d / var).sum(1).sum(0)

    # inverse multiquadratics
    def inverse_multiquadratics(d, var=16.):
        """

        :param d: (num_samples x, num_samples y)
        :param var:
        :return:
        """
        return (var / (var + d)).sum(1).sum(0)

    k = inverse_multiquadratics
    # k = gaussian
    batch_size = z_tilde.size(0)
    ce = categorical_crossentropy(weights, chorale).mean()
    zz_ker = compute_kernel(z, z, k)
    z_tilde_z_tilde_ker = compute_kernel(z_tilde, z_tilde, k)
    z_z_tilde_ker = compute_kernel(z, z_tilde, k)

    first_coefs = 1. / (batch_size * (batch_size - 1)) / 2
    second_coef = 2 / (batch_size * batch_size)
    mmd = coeff * (first_coefs * zz_ker
                   + first_coefs * z_tilde_z_tilde_ker
                   - second_coef * z_z_tilde_ker)
    return ce + mmd, ce, mmd


def mmd_reg(z_tilde, z):
    # gaussian
    def gaussian(d, var=16.):
        return torch.exp(- d / var).sum(1).sum(0)

    # inverse multiquadratics
    def inverse_multiquadratics(d, var=16.):
        """

        :param d: (num_samples x, num_samples y)
        :param var:
        :return:
        """
        return (var / (var + d)).sum(1).sum(0)

    k = inverse_multiquadratics
    # k = gaussian
    batch_size = z_tilde.size(0)
    zz_ker = compute_kernel(z, z, k)
    z_tilde_z_tilde_ker = compute_kernel(z_tilde, z_tilde, k)
    z_z_tilde_ker = compute_kernel(z, z_tilde, k)

    first_coefs = 1. / (batch_size * (batch_size - 1)) / 2
    second_coef = 2 / (batch_size * batch_size)
    mmd = (first_coefs * zz_ker
           + first_coefs * z_tilde_z_tilde_ker
           - second_coef * z_z_tilde_ker)
    return mmd


def chorale_accuracy(value, target):
    """
    :param value: list of (batch_size, chorale_length, num_notes)
    :param target: (batch_size, num_voices, chorale_length)
    :return:
    """
    batch_size, num_voices, chorale_length = target.size()
    batch_size, chorale_length, _ = value[0].size()
    num_voices = len(value)

    # put num_voices first
    target = target.transpose(0, 1)

    sum = 0
    for voice, voice_target in zip(value, target):
        max_values, max_indexes = torch.max(voice, dim=2, keepdim=False)
        num_correct = (max_indexes == voice_target).float().mean().item()
        sum += num_correct

    return sum / num_voices


def dict_pretty_print(d, endstr='\n'):
    for key, value in d.items():
        print(f'{key.capitalize()}: {value:.6}', end=endstr)


def image_to_visdom(image_location,
                    vis,
                    window,
                    legend=''):
    im = Image.open(image_location)
    width, height = im.size
    np_frame = np.array(im.getdata()).reshape((height, width, 3)) / 256
    np_frame = np_frame.transpose(2, 0, 1)
    window = vis.image(img=np_frame,
                       opts={'legend': f'{legend}'},
                       win=window
                       )
    return window
