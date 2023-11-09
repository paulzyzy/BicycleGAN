In training process, check all the parameters at the beginning and make sure using the same parameters in the inference part. The results for Tensorboard will be stored
in bicyclegan_experiment_1, the results during the training will be stored in image_results. In inference part, select generator version in the checkpoints and change the
batch size to get different number of images. Change the num_styles can give you different number of styles for same image.
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