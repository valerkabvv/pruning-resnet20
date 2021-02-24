import torch
from torchvision.datasets import CIFAR10

from models.resnet import resnet20
from data.CIFAR10_transforms import cifar_transform_train, cifar_transform_test
from training_utils.training import train

dataset_train = CIFAR10(download = True, root = './', transform = cifar_transform_train)
dataset_test = CIFAR10(train = False, download = True, root = './', transform = cifar_transform_test)

model = resnet20()

train(model, dataset_test, dataset_train,128, 100)