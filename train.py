import numpy as np
import warnings
warnings.filterwarnings("ignore")
from torch.utils import data
from torch import nn, optim
from vis_tools import *
from datasets import *
from models import *
from Utilities import *
import argparse, os
#import itertools
import torch
import time
#import pdb
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

# Create a summary writer
writer = SummaryWriter('runs/bicyclegan_experiment_1')

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Training Configurations 
# (You may put your needed configuration here. Please feel free to add more or use argparse. )
train_img_dir = './edges2shoes/train/'
checkpoints_dir = './checkpoints'
os.makedirs(checkpoints_dir, exist_ok=True)

dataset_name = "edges2shoes" 
img_shape = (3, 128, 128) # Please use this image dimension faster training purpose
num_epochs =  10
batch_size = 8
lr_rate = 0.0002 	      # Adam optimizer learning rate
betas = 0.5		  # Adam optimizer beta 1, beta 2
lambda_pixel = 10       # Loss weights for pixel loss
lambda_latent = 0.5      # Loss weights for latent regression 
lambda_kl = 0.01          # Loss weights for kl divergence
latent_dim = 8        # latent dimension for the encoded images from domain B
ndf = 64 # number of discriminator filters
# gpu_id = 
init_type='normal'
init_gain=0.02
netG='unet_128'
netD='basic_128'
norm='batch'
nl='relu'
use_dropout=False
where_add='input'
upsample='bilinear'
num_generator_filters = 64
output_nc=3	

# Random seeds (optional)
torch.manual_seed(1); np.random.seed(1)

# Create the dataset
train_dataset = Edge2Shoe(train_img_dir)
train_loader = data.DataLoader(train_dataset, batch_size=batch_size)

# Loss functions
mae_loss = torch.nn.L1Loss().to(device)

# Define generator, encoder and discriminators
generator = Generator(latent_dim, img_shape,output_nc, num_generator_filters, netG, norm, nl,
             use_dropout, init_type, init_gain, where_add, upsample).to(device)
encoder = Encoder(latent_dim).to(device)
D_VAE = Discriminator(img_shape, ndf, netD, norm, nl, init_type, init_gain, num_Ds=1).to(device)
D_LR = Discriminator(img_shape, ndf, netD, norm, nl, init_type, init_gain, num_Ds=1).to(device)

# Define optimizers for networks
optimizer_E = torch.optim.Adam(encoder.parameters(), lr=lr_rate, betas=(betas,0.999))
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_rate, betas=(betas,0.999))
optimizer_D_VAE = torch.optim.Adam(D_VAE.parameters(), lr=lr_rate, betas=(betas,0.999))
optimizer_D_LR = torch.optim.Adam(D_LR.parameters(), lr=lr_rate, betas=(betas,0.999))

# For adversarial loss
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# For adversarial loss (optional to use)
criterion_GAN = torch.nn.MSELoss().to(device)
criterion_pixel = torch.nn.L1Loss().to(device)
criterion_latent = torch.nn.L1Loss().to(device)
criterion_kl = torch.nn.KLDivLoss().to(device)

# Initialize a counter for the total number of iterations
global_step = 0
# Training
total_steps = len(train_loader)*num_epochs; step = 0

for e in range(num_epochs):
	start = time.time()
	for idx, data in enumerate(train_loader):
		loss_G = 0; loss_D_VAE = 0; loss_D_LR = 0

        # Increment the global step counter
		global_step += 1

		########## Process Inputs ##########
		edge_tensor, rgb_tensor = data
		edge_tensor, rgb_tensor = Normalize(edge_tensor).to(device), Normalize(rgb_tensor).to(device)
		real_A = edge_tensor;real_B = rgb_tensor

		b_size = real_B.size(0)
		noise = Variable(Tensor(np.random.normal(0, 1, (real_A.size(0), latent_dim))))

		#-------------------------------
		#  Train Generator and Encoder
		#------------------------------
		encoder.train(); generator.train()
		optimizer_E.zero_grad(); optimizer_G.zero_grad()

		mean, log_var = encoder(real_B)
		z = reparameterization(mean, log_var)
		# KL loss
		kl_loss = criterion_kl(z,noise)

		#generator loss for VAE-GAN
		loss_VAE_GAN, fake_B_VAE = loss_generator(generator, real_A, z, D_VAE, criterion_GAN)

		#generator loss for LR-GAN
		loss_LR_GAN, fake_B_LR = loss_generator(generator, real_A, noise, D_LR, criterion_GAN)


        #l1 loss between generated image and real image
		l1_image = loss_image(real_B, fake_B_VAE, criterion_pixel)

		loss_GE = loss_VAE_GAN + loss_LR_GAN + lambda_pixel*l1_image + lambda_kl*kl_loss

		loss_GE.backward(retain_graph=True)

		#latent loss between encoded z and noise
		l1_latent = loss_latent(fake_B_LR, encoder, noise, criterion_latent)

		loss_G = lambda_latent*l1_latent

		loss_G.backward()
        # Update G
		optimizer_G.step()
		# Update E
		optimizer_E.step()

		#----------------------------------
		#  Train Discriminator (cVAE-GAN)
		#----------------------------------

		D_VAE.train()
		optimizer_D_VAE.zero_grad()
		#loss for D_VAE
		loss_D_VAE = loss_discriminator(D_VAE, fake_B_VAE, real_B, criterion_GAN)
		loss_D_VAE.backward()
		optimizer_D_VAE.step()

		#---------------------------------
		#  Train Discriminator (cLR-GAN)
		#---------------------------------
		D_LR.train()
		optimizer_D_LR.zero_grad()
		#loss for D_LR
		loss_D_LR = loss_discriminator(D_LR, fake_B_LR, real_B, criterion_GAN)
		loss_D_LR.backward()
		optimizer_D_LR.step()

		# Log losses to TensorBoard
		writer.add_scalar('Loss/G', loss_G.item(), global_step=global_step)
		writer.add_scalar('Loss/D_VAE', loss_D_VAE.item(), global_step=global_step)
		writer.add_scalar('Loss/D_LR', loss_D_LR.item(), global_step=global_step)

		""" Optional TODO: 
			1. You may want to visualize results during training for debugging purpose
			2. Save your model every few iterations
		"""
		save_path='/home/eddieshen/CIS680/final/BicycleGAN/image_results/'
		if idx % 1000 == 0:  # visualize every 1000 batches
			visualize_images(
    			Denormalize(fake_B_VAE.detach()).cpu(), 
    			'Comparison VAE', e, idx, save_path
			)
			visualize_images(
    			Denormalize(fake_B_LR.detach()).cpu(), 
    			'Comparison LR', e, idx, save_path
			)
			visualize_images(
				Denormalize(real_B.detach()).cpu(), 
				'Real Images',e, idx, save_path
			)
			visualize_images(
				Denormalize(real_A.detach()).cpu(), 
				'Edge Images',e, idx, save_path
			)

		if idx % 500 == 0:  # save every 500 batches
			torch.save(generator.state_dict(), os.path.join(checkpoints_dir, f'generator_epoch{e}_batch{idx}.pth'))
			torch.save(encoder.state_dict(), os.path.join(checkpoints_dir, f'encoder_epoch{e}_batch{idx}.pth'))
			torch.save(D_VAE.state_dict(), os.path.join(checkpoints_dir, f'D_VAE_epoch{e}_batch{idx}.pth'))
			torch.save(D_LR.state_dict(), os.path.join(checkpoints_dir, f'D_LR_epoch{e}_batch{idx}.pth'))

	print(f'Epoch [{e+1}/{num_epochs}], Step [{global_step}], Loss G: {loss_G.item()}, Loss D_VAE: {loss_D_VAE.item()}, Loss D_LR: {loss_D_LR.item()}')

		