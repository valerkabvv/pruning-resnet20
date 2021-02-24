from torchvision import transforms

cifar_transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

cifar_transform_test = transforms.Compose([
    transforms.ToTensor()
])