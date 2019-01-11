import torch

from torchvision import datasets, transforms

import numpy as np

from VarNet.helpers import cuda_variable


class MNISTDataset:
    def __init__(self):
        self.dataset_train = datasets.MNIST('../data/mnist',
                                            train=True,
                                            download=True,
                                            transform=transforms.ToTensor())
        self.dataset_test = datasets.MNIST('../data/mnist',
                                           train=False,
                                           transform=transforms.ToTensor())

    def data_loaders(self, batch_size):
        kwargs = {'num_workers': 2, 'pin_memory': True}
        generator_train = torch.utils.data.DataLoader(self.dataset_train,
                                                      batch_size=batch_size,
                                                      shuffle=True, **kwargs)
        generator_val = torch.utils.data.DataLoader(self.dataset_test,
                                                    batch_size=batch_size,
                                                    shuffle=True, **kwargs)
        generator_test = None
        return (generator_train,
                generator_val,
                generator_test)


def normalize_images(images):
    """
    Normalize image values.
    """
    return images.float().div_(255.0).mul_(2.0).add_(-1)


class DataSampler(object):

    def __init__(self, images, attributes, batch_size, v_flip=False, h_flip=True):
        """
        Initialize the data sampler with training data.
        """
        assert images.size(0) == attributes.size(0), (images.size(), attributes.size())
        self.images = images
        self.attributes = attributes
        self.batch_size = batch_size
        self.v_flip = v_flip
        self.h_flip = h_flip

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        return self.images.size(0)

    def __iter__(self):
        for _ in range(1024):
            yield self.train_batch(32)

    def train_batch(self, bs):
        """
        Get a batch of random images with their attributes.
        """
        # image IDs
        idx = torch.LongTensor(bs).random_(len(self.images))

        # select images / attributes
        batch_x = normalize_images(self.images.index_select(0, idx).cuda())
        batch_y = self.attributes.index_select(0, idx).cuda()

        # data augmentation
        if self.v_flip and np.random.rand() <= 0.5:
            batch_x = batch_x.index_select(2,
                                           torch.arange(batch_x.size(2) - 1, -1, -1).long().cuda())
        if self.h_flip and np.random.rand() <= 0.5:
            batch_x = batch_x.index_select(3,
                                           torch.arange(batch_x.size(3) - 1, -1, -1).long().cuda())

        return cuda_variable(batch_x), cuda_variable(batch_y)

    def eval_batch(self, i, j):
        """
        Get a batch of images in a range with their attributes.
        """
        assert i < j
        batch_x = normalize_images(self.images[i:j].cuda())
        batch_y = self.attributes[i:j].cuda()
        return cuda_variable(batch_x, volatile=True), cuda_variable(batch_y, volatile=True)
