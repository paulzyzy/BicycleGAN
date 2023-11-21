import numpy as np
import warnings
warnings.filterwarnings("ignore")
import torch
from torch import nn, optim
from vis_tools import *
from datasets import *
from models import *
from Utilities import *
import  os
#import itertools
import time
#import pdb
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@hydra.main(version_base=None,config_path="config", config_name="train")
def train(cfg):
	# Random seeds (optional)
	torch.manual_seed(1); np.random.seed(1)
	torch.backends.cudnn.deterministic = True

	# save_results_path = os.path.join(os.path.join(cfg.paths.root_dir, "image_results"),cfg.experiment_name)
	# os.makedirs(save_pth_path, exist_ok=True)
	save_results_path = os.path.join(cfg.paths.root_dir, "image_results")
	os.makedirs(save_results_path, exist_ok=True)
	save_pth_path = os.path.join(cfg.paths.checkpoints_dir,cfg.experiment_name)
	os.makedirs(save_pth_path, exist_ok=True)

	#OmegaConf.resolve(cfg)
	#print(OmegaConf.to_yaml(cfg))
	model = instantiate(cfg.model.init)
	writer = SummaryWriter(cfg.experiment_path)
	# Create the dataset
	train_dataset = instantiate(cfg.datas.datasets)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.params.batch_size)

	generator = model.generator.to(device)
	encoder = model.encoder.to(device)
	D_VAE = model.D_VAE.to(device)
	D_LR = model.D_LR.to(device)

	optimizer_E = torch.optim.Adam(encoder.parameters(), lr=cfg.optimizers.lr, betas=cfg.optimizers.betas)
	optimizer_G = torch.optim.Adam(generator.parameters(), lr=cfg.optimizers.lr, betas=cfg.optimizers.betas)
	optimizer_D_VAE = torch.optim.Adam(D_VAE.parameters(), lr=cfg.optimizers.lr, betas=cfg.optimizers.betas)
	optimizer_D_LR = torch.optim.Adam(D_LR.parameters(), lr=cfg.optimizers.lr, betas=cfg.optimizers.betas)

	# For adversarial loss
	Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

	# For adversarial loss (optional to use)
	criterion_GAN = torch.nn.MSELoss().to(device)
	criterion_pixel = torch.nn.L1Loss().to(device)
	criterion_latent = torch.nn.L1Loss().to(device)
	criterion_kl = torch.nn.KLDivLoss().to(device)

	# Initialize a counter for the total number of iterations
	global_step = 0

	for e in range(cfg.params.num_epochs):
		start = time.time()
		for idx, data in enumerate(train_loader):
			# Increment the global step counter
			global_step += 1

			########## Process Inputs ##########
			edge_tensor, rgb_tensor = data
			edge_tensor, rgb_tensor = Normalize(edge_tensor).to(device), Normalize(rgb_tensor).to(device)
			real_A = edge_tensor;real_B = rgb_tensor

			b_size = real_B.size(0)
			noise = Variable(Tensor(np.random.normal(0, 1, (real_A.size(0), cfg.model.names.latent_dim))))

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

			loss_GE = loss_VAE_GAN + loss_LR_GAN + cfg.params.lambda_pixel*l1_image + cfg.params.lambda_kl*kl_loss

			loss_GE.backward(retain_graph=True)
			# Update E
			optimizer_E.step()

			#latent loss between encoded z and noise
			l1_latent = loss_latent(fake_B_LR, encoder, noise, criterion_latent)

			loss_G = cfg.params.lambda_latent*l1_latent

			loss_G.backward()
			# Update G
			optimizer_G.step()
			

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
			if idx % 1000 == 0:  # visualize every 1000 batches
				visualize_images(
					Denormalize(fake_B_VAE.detach()).cpu(), 
					'Comparison VAE', e, idx, save_results_path
				)
				visualize_images(
					Denormalize(fake_B_LR.detach()).cpu(), 
					'Comparison LR', e, idx, save_results_path
				)
				visualize_images(
					Denormalize(real_B.detach()).cpu(), 
					'Real Images',e, idx, save_results_path
				)
				visualize_images(
					Denormalize(real_A.detach()).cpu(), 
					'Edge Images',e, idx, save_results_path
				)

			if idx % 500 == 0:  # save every 500 batches
				torch.save(generator.state_dict(), os.path.join(save_pth_path, f'generator_epoch{e}_batch{idx}.pth'))
				torch.save(encoder.state_dict(), os.path.join(save_pth_path, f'encoder_epoch{e}_batch{idx}.pth'))
				torch.save(D_VAE.state_dict(), os.path.join(save_pth_path, f'D_VAE_epoch{e}_batch{idx}.pth'))
				torch.save(D_LR.state_dict(), os.path.join(save_pth_path, f'D_LR_epoch{e}_batch{idx}.pth'))

		print(f'Epoch [{e+1}/{cfg.params.num_epochs}], Step [{global_step}], Loss G: {loss_G.item()}, Loss D_VAE: {loss_D_VAE.item()}, Loss D_LR: {loss_D_LR.item()}')
		
	return 


if __name__ == '__main__':
	train()



		