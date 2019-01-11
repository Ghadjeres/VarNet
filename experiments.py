import click

from VarNet.varnet_mnist import VarNetMNIST
from VarNet.mnist.datasets import MNISTDataset


@click.command()
# === Model parameters
# - Latent space dimension
@click.option('--z_dim', default=16, type=int,
              help='Dimension of the latent spaces')
# - MLPs hidden size
@click.option('--hidden_size', default=256, type=int, )
# === KL and KL annealing: default is 'no annealing'
@click.option('--initial_beta_kl', default=1., type=float,
              help='Initial value for beta_kl parameter')
@click.option('--beta_kl_updates', default=2e-5, type=float,
              help='Increase of the beta_kl parameter after each batch')
@click.option('--burn_in_beta_kl', default=1e5, type=int,
              help='Number of epochs before increasing beta_kl')
# === MMD
@click.option('--beta_mmd', default=10, type=float,
              help='Scale factor for MMD loss')
# === Discriminator
@click.option('--beta_disc', default=10, type=float,
              help='Scale factor for Discriminator regularization')
# === Gan
@click.option('--beta_gan', default=10, type=float,
              help='Scale factor for GAN regularization')
# === Training Loading Plotting
# - training parameters -
@click.option('--train', is_flag=True)
@click.option('--batch_size', default=256, type=int,
              help='Batch size used for training')
@click.option('--num_batches', default=None, type=int,
              help='Number of batches per epoch, None for all dataset')
@click.option('--num_samples', default=1, type=int,
              help='Number of samples per input')
@click.option('--num_epochs', default=200, type=int,
              help='Number of epochs')
@click.option('--dataset_type', type=click.Choice(['mnist']),
              default='mnist'
              )
@click.option('--load', is_flag=True)
@click.option('--plot', is_flag=True)
def main(
        z_dim,
        hidden_size,
        initial_beta_kl,
        beta_kl_updates,
        burn_in_beta_kl,
        beta_mmd,
        beta_disc,
        beta_gan,
        train,
        batch_size,
        num_batches,
        num_samples,
        num_epochs,
        load,
        dataset_type,
        plot,
):
    if dataset_type == 'mnist':
        dataset = MNISTDataset()

        alpha_vae = VarNetMNIST(
            dataset=dataset,
            z_dim=z_dim,
            encoder_kwargs={'hidden_size': hidden_size},
            decoder_kwargs={'hidden_size': hidden_size},
            discriminator_kwargs={},
            attention_kwargs={}
        )

    if load:
        alpha_vae.load()

    alpha_vae.to('cuda:0')

    if train:
        alpha_vae.train_model(batch_size=batch_size,
                              num_batches=num_batches,
                              num_samples=num_samples,
                              num_epochs=num_epochs,
                              initial_beta_kl=initial_beta_kl,
                              beta_kl_update=beta_kl_updates,
                              burn_in_beta_kl=burn_in_beta_kl,
                              beta_mmd=beta_mmd,
                              beta_disc=beta_disc,
                              beta_gan=beta_gan,
                              plot=plot
                              )

if __name__ == '__main__':
    main()
