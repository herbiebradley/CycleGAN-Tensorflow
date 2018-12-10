import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import os
import numpy as np

"""Pytorch testing ground"""
dataset_id = 'facades'
project_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
path_to_dataset = os.path.join(project_dir, 'data', 'raw', dataset_id + os.sep)
testA_path = os.path.join(path_to_dataset, 'testA')
img_path = testA_path + os.sep + 'n02381460_20.jpg'

A = Image.open(img_path).convert('RGB')
A = transforms.ToTensor()(A)

# Things to try: ngf = 64, cyc_lambda = 1, 5, 20, identity loss, don't divide D loss by 2
