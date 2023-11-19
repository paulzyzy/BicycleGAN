# imports
# torch and friends
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# standard
import os
import time
import numpy as np
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import matplotlib
import hydra

from datasets import *
from Utilities import *
from models import *

matplotlib.use('Agg')
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_soft_intro_vae_toy():
    n_iter=30000
    num_vae=2000
    save_interval=5000
    recon_loss_type="mse"
    beta_kl=1.0
    beta_rec=1.0
    beta_neg=1.0
    test_iter=5000
    seed=99
    gamma_r=1e-8
    img_shape = (3, 128, 128) # Please use this image dimension faster training purpose
    batch_size = 512
    lr_e=2e-4
    lr_d=2e-4
    latent_dim = 8        # latent dimension for the encoded images from domain B
    init_type='normal'
    init_gain=0.02
    netG='unet_128'
    norm='batch'
    nl='relu'
    use_dropout=False
    where_add='input'
    upsample='bilinear'
    num_generator_filters = 64
    output_nc=3	
    train_img_dir = './edges2shoes/train/'
    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        print("random seed: ", seed)

    # --------------build models -------------------------
    train_dataset = Edge2Shoe(train_img_dir)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size)

    model = SoftIntroVAESimple(latent_dim, img_shape,output_nc, num_generator_filters, netG, norm, nl,
             use_dropout, init_type, init_gain, where_add, upsample).to(device)
    print(model)

    optimizer_e = optim.Adam(model.encoder.parameters(), lr=lr_e)
    optimizer_d = optim.Adam(model.decoder.parameters(), lr=lr_d)

    milestones = (10000, 15000)
    e_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_e, milestones=milestones, gamma=0.1)
    d_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_d, milestones=milestones, gamma=0.1)

    start_time = time.time()
    dim_scale = 0.5  # normalizing factor, 's' in the paper

    # Create the iterator outside the loop
    data_iterator = iter(train_loader)

    for it in tqdm(range(n_iter)):
        try:
            # Fetch the next batch
            batch = next(data_iterator)
        except StopIteration:
            # Reinitialize the iterator once all batches are exhausted
            data_iterator = iter(train_loader)
            batch = next(data_iterator)
        
        edge_tensor, rgb_tensor = batch
        edge_tensor, rgb_tensor = Normalize(edge_tensor).to(device), Normalize(rgb_tensor).to(device)
        real_A = edge_tensor; real_B = rgb_tensor

        # save models
        if it % save_interval == 0 and it > 0:
            save_epoch = (it // save_interval) * save_interval
            save_checkpoint(model, save_epoch, it, '')

        model.train()
        # --------------train----------------
        if it < num_vae:
            # vanilla VAE training, optimizeing the ELBO for both encoder and decoder
            # =========== Update E, D ================
            real_mu, real_logvar, z, rec = model(real_A ,real_B)

            loss_rec = calc_reconstruction_loss(real_B, rec, loss_type=recon_loss_type, reduction="mean")
            loss_kl = calc_kl(real_logvar, real_mu, reduce="mean")
            loss = beta_rec * loss_rec + beta_kl * loss_kl

            optimizer_e.zero_grad()
            optimizer_d.zero_grad()
            loss.backward()
            optimizer_e.step()
            optimizer_d.step()

            if it % test_iter == 0:
                info = "\nIter: {}/{} : time: {:4.4f}: ".format(it, n_iter, time.time() - start_time)
                info += 'Rec: {:.4f}, KL: {:.4f} '.format(loss_rec.data.cpu(), loss_kl.data.cpu())
                print(info)
        else:
            # soft-intro-vae training
            # generate random noise to produce 'fake' later
            noise_batch = torch.randn(size=(batch_size, latent_dim)).to(device)

            # =========== Update E ================
            for param in model.encoder.parameters():
                param.requires_grad = True
            for param in model.decoder.parameters():
                param.requires_grad = False

            # generate 'fake' data
            fake = model.sample(real_A,noise_batch)
            # optimize for real data
            real_mu, real_logvar = model.encode(real_B)
            z = reparameterization(real_mu, real_logvar)
            rec = model.decoder(real_A,z)  # reconstruction
            # we also want to see what is the reconstruction error from mu
            _, _, _, rec_det = model(real_A, real_B, deterministic=True)

            loss_rec = calc_reconstruction_loss(real_B, rec, loss_type=recon_loss_type, reduction="mean")
            # reconstruction error from mu (not optimized, only to observe) not used in back back propagation
            loss_rec_det = calc_reconstruction_loss(real_B, rec_det.detach(), loss_type=recon_loss_type,
                                                    reduction="mean")

            # KLD loss for the real data
            lossE_real_kl = calc_kl(real_logvar, real_mu, reduce="mean")

            # prepare the fake data for the expELBO
            fake_mu, fake_logvar, z_fake, rec_fake = model(real_A,fake.detach())
            # we also consider the reconstructions as 'fake' data, as they are output of the decoder
            rec_mu, rec_logvar, z_rec, rec_rec = model(real_A,rec.detach())

            # KLD loss for the fake data
            fake_kl_e = calc_kl(fake_logvar, fake_mu, reduce="none")
            rec_kl_e = calc_kl(rec_logvar, rec_mu, reduce="none")

            # reconstruction loss for the fake data
            loss_fake_rec = calc_reconstruction_loss(fake, rec_fake, loss_type=recon_loss_type, reduction="none")
            loss_rec_rec = calc_reconstruction_loss(rec, rec_rec, loss_type=recon_loss_type, reduction="none")

            # expELBO
            exp_elbo_fake = (-2 * dim_scale * (beta_rec * loss_fake_rec + beta_neg * fake_kl_e)).exp().mean()
            exp_elbo_rec = (-2 * dim_scale * (beta_rec * loss_rec_rec + beta_neg * rec_kl_e)).exp().mean()

            # total loss
            lossE = dim_scale * (beta_kl * lossE_real_kl + beta_rec * loss_rec) + 0.25 * (exp_elbo_fake + exp_elbo_rec)

            optimizer_e.zero_grad()
            lossE.backward()
            optimizer_e.step()

            # ========= Update D ==================
            for param in model.encoder.parameters():
                param.requires_grad = False
            for param in model.decoder.parameters():
                param.requires_grad = True

            # generate fake
            fake = model.sample(noise_batch)
            rec = model.decoder(real_A,z.detach())
            # ELBO loss for real -- just the reconstruction, KLD for real doesn't affect the decoder
            loss_rec = calc_reconstruction_loss(real_B, rec, loss_type=recon_loss_type, reduction="mean")

            # prepare fake data for ELBO
            rec_mu, rec_logvar = model.encode(rec)
            z_rec = reparameterization(rec_mu, rec_logvar)

            fake_mu, fake_logvar = model.encode(fake)
            z_fake = reparameterization(fake_mu, fake_logvar)

            rec_rec = model.decode(real_A,z_rec.detach())
            rec_fake = model.decode(real_A,z_fake.detach())

            loss_rec_rec = calc_reconstruction_loss(rec.detach(), rec_rec, loss_type=recon_loss_type, reduction="mean")
            loss_rec_fake = calc_reconstruction_loss(fake.detach(), rec_fake, loss_type=recon_loss_type,
                                                     reduction="mean")

            fake_kl = calc_kl(fake_logvar, fake_mu, reduce="mean")
            rec_kl = calc_kl(rec_logvar, rec_mu, reduce="mean")

            lossD = beta_rec * loss_rec + 0.5 * beta_kl * (fake_kl + rec_kl) + \
                    gamma_r * 0.5 * beta_rec * (loss_rec_rec + loss_rec_fake)
            lossD = dim_scale * lossD

            optimizer_d.zero_grad()
            lossD.backward()
            optimizer_d.step()

            if it % test_iter == 0:
                info = "\nIter: {}/{} : time: {:4.4f}: ".format(it, n_iter, time.time() - start_time)

                info += 'Rec: {:.4f} ({:.4f}), '.format(loss_rec.data.cpu(), loss_rec_det.data.cpu())
                info += 'Kl_E: {:.4f}, expELBO_R: {:.4f}, expELBO_F: {:.4f}, '.format(lossE_real_kl.data.cpu(),
                                                                                      exp_elbo_rec.data.cpu(),
                                                                                      exp_elbo_fake.cpu())
                info += 'Kl_F: {:.4f}, KL_R: {:.4f},'.format(fake_kl.data.cpu(), rec_kl.data.cpu())
                info += ' DIFF_Kl_F: {:.4f}'.format(-lossE_real_kl.data.cpu() + fake_kl.data.cpu())

                print(info)

            if torch.isnan(lossE) or torch.isnan(lossD):
                plt.close('all')
                raise SystemError("loss is NaN.")
        e_scheduler.step()
        d_scheduler.step()

    return model


if __name__ == '__main__':
    """
        Recommended hyper-parameters:
        - 8Gaussians: beta_kl: 0.3, beta_rec: 0.2, beta_neg: 0.9, z_dim: 2, batch_size: 512
        - 2spirals: beta_kl: 0.5, beta_rec: 0.2, beta_neg: 1.0, z_dim: 2, batch_size: 512
        - checkerboard: beta_kl: 0.1, beta_rec: 0.2, beta_neg: 0.2, z_dim: 2, batch_size: 512
        - rings: beta_kl: 0.2, beta_rec: 0.2, beta_neg: 1.0, z_dim: 2, batch_size: 512
    """
    # hyperparameters
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = train_soft_intro_vae_toy()