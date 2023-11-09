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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# Load trained generator model
generator = Generator(latent_dim, img_shape,output_nc, num_generator_filters, netG, norm, nl,
             use_dropout, init_type, init_gain, where_add, upsample).to(device)
checkpoint_path = '/home/eddieshen/CIS680/final/BicycleGAN/checkpoints/generator_epoch9_batch6000.pth'
generator.load_state_dict(torch.load(checkpoint_path, map_location=device))
generator.eval()

num_styles = 10  # Number of styles to sample
num_images = 1  # Number of images to test

# Validation images dataloader
val_dir = '/home/eddieshen/CIS680/final/BicycleGAN/edges2shoes/val/'
val_dataset = Edge2Shoe(val_dir)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=num_images, shuffle=False) 
cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
# Get first batch of images
real_A, real_B = next(iter(val_loader))
real_A = Normalize(real_A).to(device)

save_path = '/home/eddieshen/CIS680/final/BicycleGAN/inference_results'
os.makedirs(save_path, exist_ok=True)

for k in range(num_styles):
    # Generate random noise
    #noise = torch.randn(real_A.size(0), 8, 1, 1, device=device)  # latent_dim should match the one used during training
    noise = Variable(Tensor(np.random.normal(0, 1, (real_A.size(0), latent_dim))))
    #noise = (noise - noise.min()) / (noise.max() - noise.min())
    # print(noise)

    # Generate images
    with torch.no_grad():
        fake_Bs = generator(real_A, noise)
    # print(fake_Bs.shape)
    # print(Denormalize(fake_Bs.detach()).cpu())

    # Denormalize and save the generated images
    for i, fake_B in enumerate(fake_Bs):
        title = f"style{k}_image{i}"
        visualize_inference(
    			Denormalize(fake_B.detach()).cpu(), 
    			'inference', k, i, save_path, title=title
			)

