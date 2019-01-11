import random
from itertools import islice

import torch
import numpy as np
from torch.distributions import Normal
from torchvision.utils import save_image

from VarNet.varnet import VarNet
from VarNet.features import CombinedFeature
from VarNet.helpers import cuda_variable, to_numpy
from VarNet.mnist.mnist_decoder import MNISTDecoder
from VarNet.mnist.mnist_encoder import MNISTNAFEncoder
from VarNet.mnist.mnist_features import MNISTSigmoidFeature, MNISTLabelFeature
from torch.nn import functional as F

from VarNet.utils import image_to_visdom


class VarNetMNIST(VarNet):
    def __init__(self,
                 dataset,
                 z_dim,
                 encoder_kwargs,
                 decoder_kwargs,
                 discriminator_kwargs,
                 attention_kwargs
                 ):
        self.dataset = dataset

        self.features = CombinedFeature([
            MNISTLabelFeature(style_token_dim=16),
            MNISTSigmoidFeature(
                num_style_tokens=1,
                style_token_dim=16,
            ),
        ])

        # Other examples of features:
        # self.features = MNISTSigmoidFeature(
        #     num_style_tokens=2,
        #     style_token_dim=16,
        # )

        # self.features = CombinedFeature([
        #     MNISTLocalSemiIndependantSigmoidFeature(
        #         num_values=10,
        #         num_style_tokens_per_label=1,
        #         style_token_dim=16
        #     ),
        #     MNISTSigmoidFeature(num_style_tokens=1,
        #                         style_token_dim=16)]
        # )
        # self.features = MNISTLocalSemiIndependantSigmoidFeature(
        #         num_values=10,
        #         num_style_tokens_per_label=1,
        #         style_token_dim=16
        #     )
        # self.features = MNISTLocalIndependantSigmoidFeature(
        #     num_values=10,
        #     num_style_tokens_per_label=1,
        #     style_token_dim=16
        # )

        super(VarNetMNIST, self).__init__(dataset=dataset,
                                          z_dim=z_dim,
                                          discriminator_kwargs=discriminator_kwargs,
                                          style_token_dim=self.features.style_token_dim
                                          )

        self.encoder_z = MNISTNAFEncoder(z_dim=z_dim, **encoder_kwargs)
        self.decoder = MNISTDecoder(z_dim=z_dim, **decoder_kwargs)

        self.init_optimizers()

    def __repr__(self):
        return 'VarNetMNIST'

    def ce(self, value, target):
        return F.binary_cross_entropy(value, target.view(-1, 784), reduction='none').sum(1)

    def accuracy(self, value, target):
        return 0.

    def visdom_plot(self, epoch_id,
                    betas,
                    monitored_quantities_train,
                    monitored_quantities_val,
                    vis,
                    visdom_windows
                    ):
        names_train = [key + '_train' for key in monitored_quantities_train]
        names_val = [key + '_val' for key in monitored_quantities_val]
        names = names_train + names_val
        Y_train = [value for key, value in monitored_quantities_train.items()]
        Y_val = [value for key, value in monitored_quantities_val.items()]
        Y = np.array([Y_train + Y_val])
        X = np.array([[epoch_id] * len(names)])
        if epoch_id == 0:
            win_line = vis.line(Y=Y, X=X,
                                opts={'legend': names},
                                # name='Training curves'
                                )
            win_image_reconstruction = None
            win_image_variation = None
            win_image_random_variation = None
        else:
            (win_line,
             win_image_reconstruction,
             win_image_variation,
             win_image_random_variation) = visdom_windows
            # os.environ['http_proxy'] = ''
            vis.line(Y=Y, X=X, update='append',
                     opts={'legend': names},
                     # name='Training curves',
                     win=win_line)

        # Image generation
        # Tests plot generation
        # Reconstruction
        save_location = f'{self.log_dir}/reconstruction_{epoch_id}.png'
        beta_kl, beta_mmd, beta_disc = betas
        self.test_reconstruction(epoch=epoch_id,
                                 beta_kl=beta_kl,
                                 beta_mmd=beta_mmd,
                                 beta_disc=beta_disc,
                                 save_location=save_location)
        # put generation image in visdom
        win_image_reconstruction = image_to_visdom(image_location=save_location,
                                                   vis=vis,
                                                   window=win_image_reconstruction)

        # variation 2D
        save_location = f'{self.log_dir}/variation_2D_{epoch_id}.png'
        self.test_variation_lines(num_elements_per_dim=20,
                                  save_location=save_location)
        # put generation image in visdom
        win_image_variation = image_to_visdom(image_location=save_location,
                                              vis=vis,
                                              window=win_image_variation)

        # random variations
        save_location = f'{self.log_dir}/random_variations_{epoch_id}.png'
        self.test_random_variations(num_variations=50,
                                    num_elements_per_dim=10,
                                    save_location=save_location)
        # put generation image in visdom
        win_image_random_variation = image_to_visdom(image_location=save_location,
                                                     vis=vis,
                                                     window=win_image_random_variation)

        visdom_windows = (win_line,
                          win_image_reconstruction,
                          win_image_variation,
                          win_image_random_variation)
        return visdom_windows

    def test_reconstruction(self, epoch,
                            beta_kl,
                            beta_mmd,
                            beta_disc,
                            batch_size=256,
                            save_location=None):
        # todo remove betas
        self.eval()
        _, generator_test, _ = self.dataset.data_loaders(batch_size)
        with torch.no_grad():
            for i, (x, m) in enumerate(generator_test):
                x = cuda_variable(x)
                m = cuda_variable(m)
                recon_batch = self.forward(x,
                                           m=m,
                                           beta_kl=beta_kl,
                                           beta_mmd=beta_mmd,
                                           beta_disc=beta_disc,
                                           num_samples=1,
                                           train=False)['samples']
                if i == 0:
                    n = min(x.size(0), 8)
                    comparison = torch.cat([x[:n],
                                            recon_batch.view(batch_size, 1, 28, 28)[:n]])
                    save_image(comparison.cpu(),
                               save_location, nrow=n)

    def test_visualization(self,
                           num_elements_per_dim=20,
                           num_curves=6,
                           save_location=None
                           ):
        self.eval()
        batch_size = 1
        _, generator_test, _ = self.dataset.data_loaders(batch_size)

        generator_test_it = generator_test.__iter__()
        for _ in range(random.randint(0, len(generator_test))):
            x, m = next(generator_test_it)

        z_trajectory = []
        img_trajectory = []
        x_original = cuda_variable(x)
        x = cuda_variable(x.repeat(num_curves, 1, 1, 1))
        m = cuda_variable(m.repeat(num_curves))
        with torch.no_grad():
            print(self.features.alpha(x, m))
            z_star, _, _, _ = self.encoder_z.forward(x.detach())
            alpha_it = self.features.alpha_iterator(
                max_num_dimensions=2,
                num_elements_per_dim=num_elements_per_dim)

            # original
            style_vector = self.features.forward(x, m)
            logdet = cuda_variable(torch.zeros(batch_size))
            z, _, _, _ = self.z_star_to_z.forward(z_star=z_star,
                                                  logdet=logdet,
                                                  style_vector=style_vector
                                                  )
            for curve_index, coords in enumerate(z):
                original_line = torch.cat([
                    coords,
                    cuda_variable(torch.Tensor([curve_index])),
                    cuda_variable(torch.Tensor([-1])),  # -1 for original
                ], 0)
                z_trajectory.append(to_numpy(original_line[None, :]))

            # variation curves
            for alpha_index, alpha in enumerate(alpha_it):
                style_vector = self.features.style_vector_from_alpha(alpha=alpha)

                logdet = cuda_variable(torch.zeros(batch_size))
                z, _, _, _ = self.z_star_to_z.forward(z_star=z_star,
                                                      logdet=logdet,
                                                      style_vector=style_vector
                                                      )
                x_pred, samples = self.decoder.forward(z)
                img_trajectory.append(x_pred)

                for curve_index, coords in enumerate(z):
                    line = torch.cat([
                        coords,
                        cuda_variable(torch.Tensor([curve_index])),
                        cuda_variable(torch.Tensor([alpha_index]))
                    ], 0)

                    z_trajectory.append(to_numpy(line[None, :]))

        img_trajectory = torch.cat([
            t.view(num_curves, 1, 28, 28)
            for t in img_trajectory
        ], 0)
        img_trajectory = torch.cat([img_trajectory,
                                    x_original.view(1, 1, 28, 28)], 0)

        save_image(img_trajectory.cpu(),
                   'results/img_trajectories_same_x.png',
                   nrow=num_curves)

        z_trajectory = np.concatenate(z_trajectory, axis=0)
        np.savetxt('results/trajectories_same_x.csv', z_trajectory, delimiter=',')

        batch_size = num_curves
        _, generator_test, _ = self.dataset.data_loaders(batch_size)

        generator_test_it = generator_test.__iter__()
        for _ in range(random.randint(0, len(generator_test))):
            x, m = next(generator_test_it)
        trajectory = []
        img_trajectory = []
        x = cuda_variable(x)
        m = cuda_variable(m)

        with torch.no_grad():
            z_star, _, _, _ = self.encoder_z.forward(x.detach())
            alpha_it = self.features.alpha_iterator(
                max_num_dimensions=2,
                num_elements_per_dim=num_elements_per_dim)

            # original
            style_vector = self.features.forward(x, m)
            logdet = cuda_variable(torch.zeros(batch_size))
            z, _, _, _ = self.z_star_to_z.forward(z_star=z_star,
                                                  logdet=logdet,
                                                  style_vector=style_vector
                                                  )
            for curve_index, coords in enumerate(z):
                original_line = torch.cat([
                    coords,
                    cuda_variable(torch.Tensor([curve_index])),
                    cuda_variable(torch.Tensor([-1])),  # -1 for original
                ], 0)
                trajectory.append(to_numpy(original_line[None, :]))
            for alpha_index, alpha in enumerate(alpha_it):
                style_vector = self.features.style_vector_from_alpha(alpha=alpha)

                logdet = cuda_variable(torch.zeros(batch_size))
                z, _, _, _ = self.z_star_to_z.forward(z_star=z_star,
                                                      logdet=logdet,
                                                      style_vector=style_vector
                                                      )
                x_pred, samples = self.decoder.forward(z)
                img_trajectory.append(x_pred)

                for curve_index, coords in enumerate(z):
                    line = torch.cat([
                        coords,
                        cuda_variable(torch.Tensor([curve_index])),
                        cuda_variable(torch.Tensor([alpha_index]))
                    ], 0)

                    trajectory.append(to_numpy(line[None, :]))

        img_trajectory = torch.cat([
            t.view(num_curves, 1, 28, 28)
            for t in img_trajectory
        ], 0)
        img_trajectory = torch.cat([img_trajectory,
                                    x.view(batch_size, 1, 28, 28)], 0)
        save_image(img_trajectory.cpu(),
                   'results/img_trajectories.png',
                   nrow=num_curves)

        trajectory = np.concatenate(trajectory, axis=0)
        np.savetxt('results/trajectories.csv', trajectory, delimiter=',')

    def test_variation_lines(self,
                             num_elements_per_dim=20,
                             save_location=None):
        self.eval()
        batch_size = 1
        _, generator_test, _ = self.dataset.data_loaders(batch_size)

        generator_test_it = generator_test.__iter__()
        for _ in range(random.randint(0, len(generator_test))):
            x, _ = next(generator_test_it)

        trajectory = []
        x = cuda_variable(x)
        with torch.no_grad():
            z_star, _, _, _ = self.encoder_z.forward(x.detach())
            alpha_it = self.features.alpha_iterator(
                max_num_dimensions=2,
                num_elements_per_dim=num_elements_per_dim)
            for alpha in alpha_it:
                style_vector = self.features.style_vector_from_alpha(alpha=alpha)

                logdet = cuda_variable(torch.zeros(batch_size))

                z, _, _, _ = self.z_star_to_z.forward(z_star=z_star,
                                                      logdet=logdet,
                                                      style_vector=style_vector
                                                      )

                x_pred, samples = self.decoder.forward(z)
                trajectory.append(x_pred)

        trajectory.append(x)
        trajectory = torch.cat([
            t.view(batch_size, 1, 28, 28)
            for t in trajectory
        ], 0)

        save_image(trajectory.cpu(),
                   save_location,
                   nrow=num_elements_per_dim)

    def test_random_variations(self,
                               num_variations=20,
                               num_elements_per_dim=20,
                               save_location=None):
        self.eval()
        batch_size = num_variations
        _, generator_test, _ = self.dataset.data_loaders(batch_size)

        generator_test_it = generator_test.__iter__()
        for _ in range(random.randint(1, len(generator_test))):
            x, m = next(generator_test_it)

        trajectory = []
        x = cuda_variable(x)
        m = cuda_variable(m)
        with torch.no_grad():
            z_star, _, _, _ = self.encoder_z.forward(x.detach())
            z_star = Normal(torch.zeros_like(z_star),
                            torch.ones_like(z_star)).sample()

            alpha = self.features.random_alpha(batch_size)

            # todo to remove
            # alpha = self.features.random_alpha(1)
            # alpha = alpha.repeat(batch_size, 1)
            # alpha = [alpha[0].repeat(batch_size), alpha[1].repeat(batch_size, 1)]

            style_vector = self.features.style_vector_from_alpha(alpha=alpha)

            logdet = cuda_variable(torch.zeros(batch_size))

            z, _, _, _ = self.z_star_to_z.forward(z_star=z_star,
                                                  logdet=logdet,
                                                  style_vector=style_vector
                                                  )

            x_pred, samples = self.decoder.forward(z)
            trajectory.append(x_pred)

        # trajectory.append(x)
        trajectory = torch.cat([
            t.view(batch_size, 1, 28, 28)
            for t in trajectory
        ], 0)

        save_image(trajectory.cpu(),
                   save_location,
                   nrow=num_elements_per_dim)
