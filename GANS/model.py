import torch
from torch import nn
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
# Custom Imports
from . import gen
from . import desc
from . import gen_dcgan
from . import desc_dcgan

class GAN_MLP():
  '''
  GAN based on Sequential Network Architecture
  Values:
  '''
  def __init__(self, z_dim = 64, lr = 0.00001,device = 'cpu'):
    self.z_dim = z_dim
    self.lr = lr
    self.device = device
    self.gen = gen.Generator(z_dim).to(device)
    self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=self.lr)
    self.disc = desc.Discriminator().to(device) 
    self.disc_opt = torch.optim.Adam(self.disc.parameters(), lr=self.lr)
    self.criterion = nn.BCEWithLogitsLoss()

  def show_tensor_images(self, image_tensor, num_images=25, size=(1, 28, 28)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in a uniform grid.
    '''
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()

  def get_noise(self, n_samples):
    '''
    Function for creating noise vectors: Given the dimensions (n_samples, z_dim),
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
        n_samples: the number of samples to generate, a scalar
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    '''
    # NOTE: To use this on GPU with device='cuda', make sure to pass the device 
    return torch.randn(n_samples, self.z_dim, device = self.device)
  
  def get_gen_loss(self, n_samples):
    '''
    Return the loss of the generator given inputs.
    Parameters:
        gen: the generator model, which returns an image given z-dimensional noise
        disc: the discriminator model, which returns a single-dimensional prediction of real/fake
        criterion: the loss function, which should be used to compare 
              the discriminator's predictions to the ground truth reality of the images 
              (e.g. fake = 0, real = 1)
        num_images: the number of images the generator should produce, 
                which is also the length of the real images
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    Returns:
        gen_loss: a torch scalar loss value for the current batch
    '''
    #test fixture
    if n_samples == 0:
      print('testing')
      return
    noise_vectors = self.get_noise(n_samples)
    gen_out = self.gen(noise_vectors)
    disc_output_fake = self.disc(gen_out)
    gen_loss = self.criterion(disc_output_fake,torch.ones_like(disc_output_fake))
    return gen_loss

  def get_disc_loss(self, n_samples, real):
    '''
    Return the loss of the discriminator given inputs.
    Parameters:
        gen: the generator model, which returns an image given z-dimensional noise
        disc: the discriminator model, which returns a single-dimensional prediction of real/fake
        criterion: the loss function, which should be used to compare 
              the discriminator's predictions to the ground truth reality of the images 
              (e.g. fake = 0, real = 1)
        real: a batch of real images
        num_images: the number of images the generator should produce, 
                which is also the length of the real images
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    Returns:
        disc_loss: a torch scalar loss value for the current batch
    '''
    noise_vectors = self.get_noise(n_samples)
    gen_out = self.gen(noise_vectors)
    disc_output_fake = self.disc(gen_out.detach())
    loss_fake = self.criterion(disc_output_fake,torch.zeros_like(disc_output_fake))
    disc_output_real = self.disc(real)
    loss_real = self.criterion(disc_output_real,torch.ones_like(disc_output_real))
    disc_loss = (loss_fake+loss_real)/2
    return disc_loss


class GAN_DCGAN():
  '''
  GAN based on Sequential Network Architecture
  Values:
  '''
  def __init__(self, z_dim = 64, lr = 0.0002, beta1 = 0.5, beta2 = 0.999, device = 'cpu'):
    self.z_dim = z_dim
    self.lr = lr
    self.beta1 = beta1
    self.beta2 = beta2
    self.device = device
    self.gen_dcgan = gen_dcgan.Generator_dcgan(z_dim).to(device)
    self.gen_opt = torch.optim.Adam(self.gen_dcgan.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
    self.disc_dcgan = desc_dcgan.Discriminator_dcgan().to(device) 
    self.disc_opt = torch.optim.Adam(self.disc_dcgan.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
    self.criterion = nn.BCEWithLogitsLoss()
    self.gen_dcgan = self.gen_dcgan.apply(weights_init)
    self.disc_dcgan = self.disc_dcgan.apply(weights_init)

  def show_tensor_images(self, image_tensor, num_images=25, size=(1, 28, 28)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in a uniform grid.
    '''
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()

  def get_noise(self, n_samples):
    '''
    Function for creating noise vectors: Given the dimensions (n_samples, z_dim)
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
        n_samples: the number of samples to generate, a scalar
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    '''
    return torch.randn(n_samples, self.z_dim, device=self.device)

# Weights are initialized to the normal distribution
# with mean 0 and standard deviation 0.02
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)      