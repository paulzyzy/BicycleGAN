import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import os
from datasets import *
from models import *
from Utilities import *
from torch.autograd import Variable
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

# Device configuration
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
@hydra.main(version_base=None,config_path="config", config_name="eval")
def inference(cfg):
    save_infer_path = os.path.join(cfg.paths.inference_dir, cfg.experiment_name)
    os.makedirs(save_infer_path, exist_ok=True)
    model = instantiate(cfg.model.init)
    best_model_path = os.path.join(cfg.paths.checkpoints_dir,\
                                   cfg.experiment_name,\
                                   'generator_epoch9_batch5500.pth')

    generator = model.generator.to(device)
    generator.load_state_dict(torch.load(best_model_path, map_location=device))
    generator.eval()
    val_dataset = instantiate(cfg.datas.datasets)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.params.num_images, shuffle=False) 
    #Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    for i in range(5):
        real_A, real_B = next(iter(val_loader))
    real_A = Normalize(real_A).to(device)

    for k in range(cfg.params.num_styles):
        # Generate random noise
        noise = torch.randn(real_A.size(0), 8, 1, 1, device=device)  # latent_dim should match the one used during training
        #noise = Variable(Tensor(np.random.normal(0, 1, (real_A.size(0), cfg.model.names.latent_dim))))
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
        			'inference', k, i, save_infer_path, title=title
    			)
    return

if __name__ == '__main__':
    inference()