import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import os
from datasets import *
from models import *
from Utilities import *
from torch.autograd import Variable


# Device configuration
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def inference():
    recon_loss_type="mse"
    beta_kl=1.0
    beta_rec=1.0
    beta_neg=1.0
    test_iter=5000
    seed=99
    gamma_r=1e-8
    img_shape = (3, 128, 128) # Please use this image dimension faster training purpose
    batch_size = 16
    lr_e=2e-4
    lr_d=2e-4
    latent_dim = 8        # latent dimension for the encoded images from domain B
    init_type='normal'
    init_gain=0.02
    netG='unet_128'
    norm='batch'
    nl='relu'
    use_dropout=False
    where_add='all'
    upsample='bilinear'
    num_generator_filters = 64
    output_nc=3	
    val_img_dir = './edges2shoes/val/'
    save_infer_path = './VAE_inference/'
    num_styles = 8

    # --------------build models -------------------------
    val_dataset = Edge2Shoe(val_img_dir)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size)

    real_A, real_B = next(iter(val_loader))
    real_A = Normalize(real_A).to(device)

    model = SoftIntroVAESimple(latent_dim, img_shape,output_nc, num_generator_filters, netG, norm, nl,
             use_dropout, init_type, init_gain, where_add, upsample).to(device)

    # Load the trained model weights
    checkpoint_path = '/home/paulzy/BicycleGAN/saves/model_epoch_90000_iter_90000.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    model.eval()

    for k in range(num_styles):
        # Generate random noise
        noise = torch.randn(real_A.size(0), latent_dim, 1, 1, device=device)
        with torch.no_grad():
            # Generate images
            fake_Bs = model.decoder(real_A, noise)

        # Denormalize and save the generated images
        for i, fake_B in enumerate(fake_Bs):
            title = f"style{k}_image{i}"
            visualize_inference(
        			Denormalize(fake_B.detach()).cpu(), 
        			'inference', k, i, save_infer_path, title=title
    			)
    return

if __name__ == '__main__':
    inference()