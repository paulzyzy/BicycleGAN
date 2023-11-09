import torch
import torch.nn as nn
from torch.nn import init
import functools
import matplotlib.pyplot as plt
import torchvision
import numpy as np
import os
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt

# Normalize image tensor
def Normalize(image):
	return (image/255.0-0.5)*2.0

# Denormalize image tensor
def Denormalize(tensor):
	return ((tensor+1.0)/2.0)*255.0

def visualize_inference(denorm_tensor, prefix, style_num, image_num, save_dir,title):
    """
    Visualizes the inference by saving side-by-side images of input and output.

    Args:
        denorm_tensor (Tensor): The denormalized generated image tensor.
        prefix (str): Prefix for saving the file.
        style_num (int): The style number to include in the filename.
        image_num (int): The image number to include in the filename.
        save_dir (str): The directory to save the images.
    """
    image_path = os.path.join(save_dir, f"{prefix}_style{style_num}_image{image_num}.png")
    # Convert the tensor to PIL image for saving
    image = denorm_tensor.permute(1, 2, 0).numpy()
    image = np.clip(image, 0, 255).astype(np.uint8)
    # pil_image = F.to_pil_image(denorm_tensor)
    # pil_image.save(image_path)
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    
    # Save the figure
    plt.savefig(image_path)
    plt.close()  # Close the figure to avoid display

def visualize_images(image, title, epoch, idx, save_path):
    # Create a grid with 2 images per row for each pair of real and generated images
    grid_img = torchvision.utils.make_grid(image, nrow=2)
    # Convert to a numpy image
    np_grid_img = grid_img.permute(1, 2, 0).numpy()
    np_grid_img = np.clip(np_grid_img, 0, 255).astype(np.uint8)

    # Plot and save the images
    plt.figure(figsize=(16, 12))
    plt.imshow(np_grid_img)
    plt.title(title)
    plt.axis('off')
    
    # Save the figure
    file_name = f"{title}_epoch{epoch}_batch{idx}.png"
    save_full_path = os.path.join(save_path, file_name)
    plt.savefig(save_full_path)
    plt.close()  # Close the figure to avoid display


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_non_linearity(layer_type='relu'):
    if layer_type == 'relu':
        nl_layer = functools.partial(nn.ReLU, inplace=True)
    elif layer_type == 'lrelu':
        nl_layer = functools.partial(
            nn.LeakyReLU, negative_slope=0.2, inplace=True)
    elif layer_type == 'elu':
        nl_layer = functools.partial(nn.ELU, inplace=True)
    else:
        raise NotImplementedError(
            'nonlinearity activitation [%s] is not found' % layer_type)
    return nl_layer

def reparameterization(mean, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)#Returns a tensor with the same size as std that is filled with random numbers from a normal distribution with mean 0 and variance 1
    z = mean + eps * std

    return z

def loss_image(real_B, fake, criterion_pixel):
    loss_pixel = criterion_pixel(fake, real_B)
    return loss_pixel

def loss_latent(fake, E, noise, criterion_latent):
    encoded_z_tuple = E(fake)
    encoded_z = encoded_z_tuple[0]
    encoded_z = encoded_z.squeeze() # squeeze the dimension of 1
    loss_latent = criterion_latent(encoded_z, noise)
    return loss_latent

def loss_discriminator(D, fake, real_B, criterion):
    '''
    1. Forward real images into the discriminator
    2. Compute loss between Valid_label and dicriminator output on real images
    4. Forward fake images to the discriminator
    5. Compute loss between Fake_label and discriminator output on fake images
    6. sum real loss and fake loss as the loss_D
    7. we also need to output fake images generate by G(noise) for loss_generator computation
    '''
    
    real_output = D(real_B)
    real_output = real_output.squeeze()
    valid_label = torch.ones_like(real_output, requires_grad=False)
    real_loss = criterion(real_output, valid_label)

    fake_output = D(fake.detach())
    fake_output = fake_output.squeeze()
    fake_label = torch.zeros_like(fake_output, requires_grad=False)
    fake_loss = criterion(fake_output, fake_label)
    loss_D = real_loss + fake_loss
    # return total loss_D
    return loss_D

def loss_generator(G, real, z, D, criterion_GAN):
    '''
    loss_generator function is applied to compute loss for both generator G_AB and G_BA:
    For example, we want to compute the loss for G_AB.
    real2G will be the real image in domain A, then we map real2G into domain B to get fake B,
    then we compute the loss between D_B(fake_B) and valid, which is 1.
    The fake_B image will also be one of the outputs, since we want to use it in the loss_cycle_consis.
    '''

    # Generate fake images by forwarding real images from domain A/B through the generator G
    fake = G(real, z)

    # Forward the generated fake images through the corresponding discriminator D
    fake_pred = D(fake)
    valid = torch.ones_like(fake_pred, requires_grad=False)
    # Compute loss between the discriminator's predictions on the fake images and valid labels (which are all 1s)
    loss_G = criterion_GAN(fake_pred, valid)

    return loss_G, fake