import torch
from torchvision.datasets import CIFAR10

from models.resnet import resnet20
from data.CIFAR10_transforms import cifar_transform_test
from training_utils.training import validate

dataset_test = CIFAR10(train = False, download = True, root = './', transform = cifar_transform_test)

model = resnet20()
model.load_state_dict(torch.load('best_model.ckpt'))

valitade(model, dataset_test)