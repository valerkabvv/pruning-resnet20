import torch
from torchvision.datasets import CIFAR10

from models.resnet import resnet20
from data.CIFAR10_transforms import cifar_transform_test
from training_utils.training import validate

import sys

if __name__ == '__main__':
    
    if len(sys.argv) == 1:
        model_path = 'best_model.ckpt'
    else:
        model_path = sys.argv[1]
    
    dataset_test = CIFAR10(train = False, download = True, root = './', transform = cifar_transform_test)

    model = resnet20()
    model.load_state_dict(torch.load(model_path))

    validate(model, dataset_test)