import torch
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
from PIL import Image
import os
from datasets import *
from models import *
from Utilities import *
from torch.autograd import Variable
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from metrics import *
import itertools
import math


# Device configuration
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
@hydra.main(version_base=None, config_path="config", config_name="eval")
def inference(cfg):
    save_infer_path = os.path.join(
        cfg.paths.inference_dir, cfg.experiment_name)
    
    os.makedirs(save_infer_path, exist_ok=True)

    model = instantiate(cfg.model.init)

    best_model_path = os.path.join(cfg.paths.checkpoints_dir,
                                   cfg.experiment_name,
                                   'generator_epoch20_batch6000.pth')

    generator = model.generator.to(device)
    generator.load_state_dict(torch.load(best_model_path, map_location=device))
    generator.eval()
    val_dataset = instantiate(cfg.datas.datasets)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.params.batch_size, shuffle=False)

    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
    PerceptualLoss = instantiate(cfg.metrics.init)

    dist_list = []
    for i, data in enumerate(val_loader):
        real_A, _ = data
        real_A = Normalize(real_A).to(device)

        out_styles = []
        for k in range(cfg.params.num_styles):
            # Generate random noise
            noise = Variable(
                    Tensor(
                        np.random.normal(0, 1, (
                                                real_A.size(0),
                                                cfg.model.names.latent_dim))))

            # Generate images
            with torch.no_grad():
                fake_Bs = generator(real_A, noise)
                out_styles.append(fake_Bs)

            # Denormalize and save the generated images
            visualize_inference(
                    Denormalize(fake_Bs[0].detach()).cpu(),
                    'inference', k, i,
                    save_infer_path, title=f"style{k}_image{i}"
            )
        
        # Compute the perceptual loss
        dist = 0
        for imgs_pair in itertools.combinations(range(len(out_styles)), 2):
            index1, index2 = imgs_pair
            img1, img2 = out_styles[index1], out_styles[index2]
            perceptual_loss = PerceptualLoss(img1, img2)
            dist += perceptual_loss.item()
        dist /= (
            math.factorial(len(out_styles)) //
            (2 * math.factorial(len(out_styles) - 2))
            )
        dist_list.append(dist)
        print(f"Perceptual loss for image {i}: {dist}")
    
    print(f"Average perceptual loss: {sum(dist_list) / len(dist_list)}")
    save_infer_path = os.path.join(save_infer_path, 'metrics')
    plot_distances(dist_list, save_infer_path)

    return

if __name__ == '__main__':
    inference()