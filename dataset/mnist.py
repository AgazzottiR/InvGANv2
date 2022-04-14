import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

image_size = 28
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(size=image_size, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


def get_mnist_trainloader(batch_size=128, workers=2):
    dataset = dset.MNIST(root='./data', train=True, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
                                             drop_last=True)
    return dataloader


def get_mnist_testloader(batch_size=128, workers=2):
    dataset = dset.MNIST(root='./data', train=False, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
                                             drop_last=True)
    return dataloader

