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



if __name__ == '__main__':
    inference()