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

from datasets import *
from Utilities import *
from models import *
import torch
torch.cuda.empty_cache()

matplotlib.use('Agg')
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def train_soft_intro_vae_toy():
    pretrained = False
    checkpoint_path = '/home/paulzy/BicycleGAN/saves/model_epoch_90000_iter_90000.pth'
    start_epoch = 0
    num_epochs = 150
    num_vae=0
    visualize_epoch=10
    test_iter=1000
    save_interval=10
    seed=99

    recon_loss_type="mse"
    beta_kl=1.0
    beta_rec=1.0
    beta_neg=256

    gamma_r=1e-8
    img_shape = (3, 128, 128) # Please use this image dimension faster training purpose
    batch_size = 32
    val_size = 8
    lr_e=2e-4
    lr_d=2e-4
    latent_dim = 128        # latent dimension for the encoded images from domain B
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
    val_img_dir = './edges2shoes/val/'
    val_dataset = Edge2Shoe(val_img_dir)
    val_loader = data.DataLoader(val_dataset, batch_size=val_size)
    val_A, val_B = next(iter(val_loader))
    val_A = Normalize(val_A).to(device)
    val_results_path = './VAE_val_results/'

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
    if pretrained:
        # Load the trained model weights
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
    # print(model)

    optimizer_e = optim.Adam(model.encoder.parameters(), lr=lr_e)
    optimizer_d = optim.Adam(model.decoder.parameters(), lr=lr_d)

    e_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_e, milestones=(350,), gamma=0.1)
    d_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_d, milestones=(350,), gamma=0.1)

    start_time = time.time()
    scale = 1 / (img_shape[0] * img_shape[1]* img_shape[2])  # normalizing factor, 's' in the paper

    cur_iter = 0

    for epoch in range(start_epoch, num_epochs):
        # save models
        if epoch % save_interval == 0 and epoch > 0:
            save_epoch = (epoch // save_interval) * save_interval
            prefix = dataset + "_soft_intro_vae" + "_betas_" + str(beta_kl) + "_" + str(beta_neg) + "_" + str(
                beta_rec) + "_"
            save_checkpoint(model, save_epoch, cur_iter, prefix)

        model.train()
        batch_kls_real = []
        batch_kls_fake = []
        batch_kls_rec = []
        batch_rec_errs = []

        for it, batch in enumerate(train_loader):
            edge_tensor, rgb_tensor = batch
            edge_tensor, rgb_tensor = Normalize(edge_tensor).to(device), Normalize(rgb_tensor).to(device)
            real_A = edge_tensor; real_B = rgb_tensor
        
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
                    info = "\nEpoch[{}]({}/{}): time: {:4.4f}: ".format(epoch, it, len(train_loader),
                                                                        time.time() - start_time)
                    info += 'Rec: {:.4f}, KL: {:.4f}, '.format(loss_rec.data.cpu(), loss_kl.data.cpu())
                    print(info)
            else:
                # soft-intro-vae training
                # generate random noise to produce 'fake' later
                noise_batch = torch.randn(size=(real_A.shape[0], latent_dim)).to(device)

                # =========== Update E ================
                for param in model.encoder.parameters():
                    param.requires_grad = True
                for param in model.decoder.parameters():
                    param.requires_grad = False
                # print(real_A.shape,noise_batch.shape)
                # generate 'fake' data
                fake = model.sample(real_A,noise_batch)
                # optimize for real data
                real_mu, real_logvar = model.encode(real_B)
                z = reparameterization(real_mu, real_logvar)
                rec = model.decoder(real_A,z)  # reconstruction
                loss_rec = calc_reconstruction_loss(real_B, rec, loss_type=recon_loss_type, reduction="mean")
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
                exp_elbo_fake = (-2 * scale * (beta_rec * loss_fake_rec + beta_neg * fake_kl_e)).exp().mean()
                exp_elbo_rec = (-2 * scale * (beta_rec * loss_rec_rec + beta_neg * rec_kl_e)).exp().mean()

                # total loss
                lossE = scale * (beta_rec * loss_rec + beta_kl * lossE_real_kl) + 0.25 * (exp_elbo_fake + exp_elbo_rec)
                
                # backprop
                optimizer_e.zero_grad()
                lossE.backward()
                optimizer_e.step()

                # ========= Update D ==================
                for param in model.encoder.parameters():
                    param.requires_grad = False
                for param in model.decoder.parameters():
                    param.requires_grad = True

                # generate fake
                fake = model.sample(real_A, noise_batch)
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
                lossD = scale * lossD

                optimizer_d.zero_grad()
                lossD.backward()
                optimizer_d.step()

                if cur_iter % test_iter == 0:
                    info = "\nEpoch[{}]({}/{}): time: {:4.4f}: ".format(epoch, it, len(train_loader), 
                                                                        time.time() - start_time)
                    info += 'Rec: {:.4f}, '.format(loss_rec.data.cpu())
                    info += 'Kl_E: {:.4f}, expELBO_R: {:.4e}, expELBO_F: {:.4e}, '.format(lossE_real_kl.data.cpu(),
                                                                                    exp_elbo_rec.data.cpu(),
                                                                                    exp_elbo_fake.cpu())
                    info += 'Kl_F: {:.4f}, KL_R: {:.4f}'.format(rec_kl.data.cpu(), fake_kl.data.cpu())
                    info += ' DIFF_Kl_F: {:.4f}'.format(-lossE_real_kl.data.cpu() + fake_kl.data.cpu())
                    print(info)

                if torch.isnan(lossE) or torch.isnan(lossD):
                    plt.close('all')
                    raise SystemError("loss is NaN.")
            cur_iter += 1

        if epoch % visualize_epoch == 0:  # visualize every 10 epochs
            fake_results = model.sample_with_noise(val_A, num_samples=val_size, device=device)
            visualize_images(
                Denormalize(fake_results.detach()).cpu(), 
                'Val_fake', epoch, cur_iter, val_results_path
            )
            # visualize_images(
            #     Denormalize(real_B.detach()).cpu(), 
            #     'Real Images',epoch, cur_iter, val_results_path
            # )
            # visualize_images(
            #     Denormalize(real_A.detach()).cpu(), 
            #     'Edge Images',epoch, cur_iter, val_results_path
            # )
        e_scheduler.step()
        d_scheduler.step()

        if epoch > num_vae - 1:
            kls_real.append(np.mean(batch_kls_real))
            kls_fake.append(np.mean(batch_kls_fake))
            kls_rec.append(np.mean(batch_kls_rec))
            rec_errs.append(np.mean(batch_rec_errs))

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