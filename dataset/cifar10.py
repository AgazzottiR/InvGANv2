import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

image_size = 32
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(size=image_size, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4821, 0.4465), (0.2467, 0.2431, 0.2611)),
])


def get_cifar10_trainloader(batch_size=128, workers=2):
    dataset = dset.CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
                                             drop_last=True)
    return dataloader


def get_cifar10_testloader(batch_size=128, workers=2):
    dataset = dset.CIFAR10(root='./data', train=False, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
                                             drop_last=True)
    return dataloader

