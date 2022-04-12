import torch
from models.networks import MappingNetwork, Generator, Discriminator



def get_model_to_evaluate(dataset_name="CIFAR"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pathG = r'../checkpoints/InvGan/latest_net_G.pth' if dataset_name == "CIFAR" else r'../checkpoints/InvGAN_mnist/latest_net_G.pth'
    pathD = r'../checkpoints/InvGan/latest_net_D.pth' if dataset_name == "CIFAR" else r'../checkpoints/InvGAN_mnist/latest_net_D.pth'
    pathM = r'../checkpoints/InvGan/latest_net_M.pth' if dataset_name == "CIFAR" else r'../checkpoints/InvGAN_mnist/latest_net_M.pth'

    checkepoint_gen = torch.load(pathG)
    checkepoint_disc = torch.load(pathD)
    checkpoint_mn = torch.load(pathM)

    netMN = MappingNetwork().to(device)
    netMN.load_state_dict(checkpoint_mn)
    netMN.eval()
    netG = Generator(output_nc=1 if dataset_name == "MNIST" else 3)
    netG.to(device)
    netG.load_state_dict(checkepoint_gen)
    netG.eval()
    netD = Discriminator(ic=1 if dataset_name == "MNIST" else 3)
    netD.to(device)
    netD.load_state_dict(checkepoint_disc)
    netD.eval()


    return netMN, netG, netD
