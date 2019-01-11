import os

from itertools import islice
import torch
from VarNet.torchkit.torchkit import utils


from torch.distributions import Normal

from tqdm import tqdm

from VarNet.discriminator import Discriminator
from VarNet.helpers import cuda_variable, free_params, frozen_params
from VarNet.naf import NAF
from VarNet.utils import mmd_reg, dict_pretty_print


class VarNet:
    """ Abstract class
    Superclasses must implement encoder_z, features and decoder
    """

    def __init__(self,
                 dataset,
                 z_dim,
                 style_token_dim,
                 discriminator_kwargs,
                 ):
        self.dataset = dataset
        self.bits = 0.1
        self.style_token_dim = style_token_dim
        self.z_dim = z_dim

        self.encoder_z = None
        self.decoder = None

        self.z_star_to_z = NAF(style_token_dim=self.style_token_dim,
                               z_dim=self.z_dim)

        self.disc_z = Discriminator(z_dim=self.z_dim,
                                    num_style_tokens=self.style_token_dim,
                                    **discriminator_kwargs)

        # self.features = None
        # self.optimizer = None
        # self.optimizer_features = None
        # self.optimizer_naf = None
        # self.optimizer_disc = None

    def init_optimizers(self):
        """
        Must be called in all subclasses
        :return:
        """
        self.optimizer = torch.optim.Adam(list(self.decoder.parameters())
                                          + list(self.encoder_z.parameters())
                                          )
        self.optimizer_features = torch.optim.Adam(
            self.features.parameters(),
            lr=1e-4
        )
        self.optimizer_naf = torch.optim.Adam(self.z_star_to_z.parameters(),
                                              lr=1e-3)
        self.optimizer_disc = torch.optim.Adam(
            self.disc_z.parameters(),
            lr=1e-4)

    def preprocessing(self, *tensors):
        x, m = tensors
        return cuda_variable(x), cuda_variable(m)

    @property
    def model_dir(self):
        if not os.path.exists('models'):
            os.mkdir('models')
        return f'models/{self.__repr__()}'

    @property
    def log_dir(self):
        if not os.path.exists('logs'):
            os.mkdir('logs')
        log_dir = f'logs/{self.__repr__()}'
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        return log_dir

    def to(self, device):
        self.decoder.to(device)
        self.encoder_z.to(device)
        self.z_star_to_z.to(device)
        self.disc_z.to(device)
        self.features.to(device)

    def save(self):
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        torch.save(self.encoder_z.state_dict(), f'{self.model_dir}/encoder')
        torch.save(self.decoder.state_dict(), f'{self.model_dir}/decoder')
        torch.save(self.z_star_to_z.state_dict(), f'{self.model_dir}/naf')
        torch.save(self.disc_z.state_dict(), f'{self.model_dir}/disc_z')
        torch.save(self.features.state_dict(), f'{self.model_dir}/features')
        print(f'Model {self.__repr__()} saved')

    def load(self):
        print(f'Loading models {self.__repr__()}')
        self.encoder_z.load_state_dict(torch.load(f'{self.model_dir}/encoder'))
        self.decoder.load_state_dict(torch.load(f'{self.model_dir}/decoder'))
        self.z_star_to_z.load_state_dict(torch.load(f'{self.model_dir}/naf'))
        self.disc_z.load_state_dict(torch.load(f'{self.model_dir}/disc_z'))
        self.features.load_state_dict(torch.load(f'{self.model_dir}/features'))

    def train(self):
        self.encoder_z.train()
        self.decoder.train()
        self.z_star_to_z.train()
        self.disc_z.train()
        self.features.train()

    def eval(self):
        self.encoder_z.eval()
        self.decoder.eval()
        self.z_star_to_z.eval()
        self.disc_z.eval()
        self.features.eval()

    def ce(self, value, target):
        raise NotImplementedError

    def accuracy(self, value, target):
        raise NotImplementedError

    # Adversarial loss  for z and z_star
    def forward_disc(self, x, m=None):
        batch_size = self.get_batch_size(x, m)

        # compute z_star (aggregated posterior)
        z_star, _, _, _ = self.encoder_z.forward(x)

        style_vector = self.features(x, m)

        random_style_vector = self.features.random_style_vector(batch_size=batch_size,
                                                                style_vector=style_vector)

        prob_false_z = self.disc_z.forward(z=z_star, alpha=style_vector)
        prob_true_z = self.disc_z.forward(z=z_star, alpha=random_style_vector)
        loss_z = torch.log(prob_true_z) + torch.log(1 - prob_false_z)

        loss = -loss_z.mean()

        monitored_quantities = dict(
            loss_disc=loss.item(),
            prob_false_z=prob_false_z.mean().item(),
            prob_true_z=prob_true_z.mean().item()
        )

        return dict(
            loss=loss,
            monitored_quantities=monitored_quantities,
            samples=None
        )

    def disc_reg(self, z_star, style_vector):
        prob_disc_z = self.disc_z.forward(z_star, style_vector)[:, 0]
        return torch.log(prob_disc_z)

    def get_batch_size(self, x, m):
        return x.size(0)

    def repeat_(self, x, m, num_samples):
        # add num_samples for chorale and metadata
        batch_size = x.size(0)
        metadata_size = m.size()
        x_size = x.size()
        x = x.unsqueeze(0).repeat(num_samples, *((1,) * len(x_size)))
        x = x.view(num_samples * batch_size, *x_size[1:])
        m = m.unsqueeze(0).expand(num_samples, *((-1,) * len(metadata_size))).contiguous()
        m = m.view(num_samples * batch_size, *metadata_size[1:])
        return x, m

    def forward(self,
                x,
                m,
                train,
                beta_kl,
                beta_mmd,
                num_samples=1,
                beta_disc=1.):
        x, m = self.repeat_(x, m, num_samples=num_samples)

        z_star, logdet_z_star, context, noise = self.encoder_z(x)

        # compute alpha_from_input
        style_vector = self.features(x, m)

        # z knowing true tokens
        z, logdet_z, _, u_z = self.z_star_to_z.forward(z_star=z_star,
                                                       logdet=logdet_z_star,
                                                       style_vector=style_vector
                                                       )

        # compute weights
        x_reconstruct, samples = self.decoder.forward(
            z=z,
            x=x,
            train=train)

        # KL on z_star
        zero = cuda_variable(torch.zeros(1))
        logqz_star = utils.log_normal(noise, zero, zero).sum(1) - logdet_z_star
        logpz_star = utils.log_normal(z_star, zero, zero).sum(1)
        kl_star = logqz_star - logpz_star

        # free bits or beta?
        # kl = 0.1 * torch.max(kl, torch.ones_like(kl) * self.bits)

        # KL on z
        zero = cuda_variable(torch.zeros(1))
        logqz = utils.log_normal(noise, zero, zero).sum(1) - logdet_z
        logpz = utils.log_normal(z, zero, zero).sum(1)
        kl = logqz - logpz

        # compute z (prior)
        prior_distribution = Normal(torch.zeros_like(z_star),
                                    torch.ones_like(z_star))
        z_prior = prior_distribution.sample()
        z_star_prior = prior_distribution.sample()

        # mmd on z star
        mmd_z_star = mmd_reg(
            z_tilde=z_star,
            z=z_star_prior
        )

        # mmd on z
        mmd_z = mmd_reg(
            z_tilde=z,
            z=z_prior
        )

        ce = self.ce(value=x_reconstruct, target=x)

        disc_reg = self.disc_reg(z_star=z_star,
                                 style_vector=style_vector)

        loss = (ce
                + beta_kl * kl_star
                # + beta_kl * kl
                + beta_mmd * mmd_z
                # + beta_mmd * mmd_z_star
                - beta_disc * disc_reg
                )

        loss = loss.mean()

        acc = self.accuracy(x_reconstruct, x)

        monitored_quantities = dict(
            loss=loss.item(),
            ce=ce.mean().item(),
            kl_star=kl_star.mean().item(),
            kl=kl.mean().item(),
            disc_reg=disc_reg.mean().item(),
            mmd_z=mmd_z.mean().item(),
            mmd_z_star=mmd_z_star.mean().item(),
            acc=acc * 100
        )

        return dict(loss=loss,
                    monitored_quantities=monitored_quantities,
                    samples=samples
                    )

    def train_model(self,
                    batch_size,
                    num_batches=None,
                    num_samples=1,
                    num_epochs=10,
                    initial_beta_kl=1.,
                    beta_kl_update=0.01,
                    burn_in_beta_kl=0,
                    beta_disc=10,
                    beta_mmd=10,
                    plot=False,
                    **kwargs
                    ):

        (generator_train,
         generator_val,
         generator_test) = self.dataset.data_loaders(batch_size=batch_size)
        beta_kl = initial_beta_kl
        if plot:
            import visdom
            vis = visdom.Visdom()
            visdom_windows = None

        for epoch_id in range(num_epochs):
            monitored_quantities_train, beta_kl = self.epoch(
                data_loader=generator_train,
                num_samples=num_samples,
                beta_kl=beta_kl,
                beta_update=beta_kl_update,
                beta_disc=beta_disc,
                beta_mmd=beta_mmd,
                train=True,
                num_batches=num_batches,
                burn_in=epoch_id < burn_in_beta_kl,
            )

            monitored_quantities_val, beta_kl = self.epoch(
                data_loader=generator_val,
                num_samples=num_samples,
                beta_kl=beta_kl,
                beta_disc=beta_disc,
                beta_mmd=beta_mmd,
                beta_update=beta_kl_update,
                train=False,
                num_batches=num_batches // 2 if num_batches is not None else None,
                burn_in=True,
            )

            print(f'======= Epoch {epoch_id} =======')
            print(f'Beta KL: {beta_kl}, Beta MMD: {beta_mmd}, Beta Disc: {beta_disc}')
            print(f'---Train---')
            dict_pretty_print(monitored_quantities_train, endstr=' ' * 5)
            print()
            print(f'---Val---')
            dict_pretty_print(monitored_quantities_val, endstr=' ' * 5)
            print('\n')

            self.save()

            if plot:
                # self.test_discrete_variations(num_variations=1,
                #                               save_location=save_location)
                betas = (beta_kl, beta_mmd, beta_disc)
                visdom_windows = self.visdom_plot(epoch_id,
                                                  betas,
                                                  monitored_quantities_train,
                                                  monitored_quantities_val,
                                                  vis,
                                                  visdom_windows,
                                                  )

                # self.test_variation_lines()
                # self.test_random_variations()

    def visdom_plot(self, epoch_id,
                    monitored_quantities_train,
                    monitored_quantities_val,
                    vis,
                    visdom_windows,
                    save_location):
        """

        :param epoch_id:
        :param monitored_quantities_train:
        :param monitored_quantities_val:
        :param vis:
        :param visdom_windows:
        :param save_location:
        :return: visdom_windows
        """
        raise NotImplementedError

    def epoch(self, data_loader,
              num_samples=1,
              beta_kl=1.,
              beta_disc=10,
              beta_update=1e-5,
              burn_in=False,
              beta_mmd=10,
              train=True,
              num_batches=None,
              ):
        if num_batches is None or num_batches > len(data_loader):
            num_batches = len(data_loader)

        means = None

        if train:
            self.train()
        else:
            self.eval()

        for sample_id, tensors in tqdm(enumerate(islice(data_loader,
                                                        num_batches))):
            if not burn_in:
                beta_kl = min(beta_kl + beta_update, 1.)

            x, m = self.preprocessing(*tensors)

            # ========Train discriminator ==============
            if train:
                free_params(self.disc_z)
                frozen_params(self.encoder_z)
                frozen_params(self.decoder)
                frozen_params(self.z_star_to_z)
                frozen_params(self.features)
                self.optimizer.zero_grad()
                self.optimizer_naf.zero_grad()
                self.optimizer_disc.zero_grad()
                self.optimizer_features.zero_grad()

                forward_pass_disc = self.forward_disc(x=x,
                                                      m=m,
                                                      )

                loss_disc = beta_disc * forward_pass_disc['loss']
                loss_disc.backward()
                torch.nn.utils.clip_grad_norm_(self.disc_z.parameters(), 5)
                self.optimizer_disc.step()
            else:
                forward_pass_disc = {'monitored_quantities': {}}

            # ========Train generator ==================

            self.optimizer.zero_grad()
            self.optimizer_naf.zero_grad()
            self.optimizer_disc.zero_grad()
            self.optimizer_features.zero_grad()

            frozen_params(self.disc_z)

            free_params(self.features)
            free_params(self.encoder_z)
            free_params(self.decoder)
            free_params(self.z_star_to_z)
            self.disc_z.eval()

            forward_pass_gen = self.forward(x,
                                            m,
                                            num_samples=num_samples,
                                            train=train,
                                            beta_kl=beta_kl,
                                            beta_mmd=beta_mmd,
                                            beta_disc=beta_disc
                                            )
            loss = forward_pass_gen['loss']

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 5)
                torch.nn.utils.clip_grad_norm_(self.encoder_z.parameters(), 5)
                torch.nn.utils.clip_grad_norm_(self.z_star_to_z.parameters(), 5)
                torch.nn.utils.clip_grad_norm_(self.features.parameters(), 5)
                self.optimizer.step()
                self.optimizer_naf.step()
                self.optimizer_features.step()

            # Monitored quantities
            monitored_quantities = dict(forward_pass_gen['monitored_quantities'],
                                        **forward_pass_disc['monitored_quantities'])
            # average quantities
            if means is None:
                means = {key: 0
                         for key in monitored_quantities}
            means = {
                key: value + means[key]
                for key, value in monitored_quantities.items()
            }

            del loss

        # last_samples = forward_pass_gen['samples']

        # renormalize monitored quantities
        means = {
            key: value / num_batches
            for key, value in means.items()
        }
        return means, beta_kl

    def test_discrete_variations(self,
                                 num_variations=2,
                                 **kwargs):
        raise NotImplementedError

    def test_variation_lines(self, **kwargs):
        raise NotImplementedError

    def test_random_variations(self, num_variations=20):
        raise NotImplementedError

    def test_aggregated_posterior(self, num_points=None):
        raise NotImplementedError
