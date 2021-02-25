def get_disc_loss(gen, disc, criterion, real, num_images, z_dim, device):
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
    noise_vectors = get_noise(num_images, z_dim, device)
    gen_out = gen(noise_vectors)
    disc_output_fake = disc(gen_out.detach())
    loss_fake = criterion(disc_output_fake,torch.zeros_like(disc_output_fake))
    disc_output_real = disc(real)
    loss_real = criterion(disc_output_real,torch.ones_like(disc_output_real))
    disc_loss = (loss_fake+loss_real)/2
    return disc_loss