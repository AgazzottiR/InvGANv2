import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import torch.nn as nn


class GanVanillaModel(BaseModel):
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_name = ['GAN', 'GAN_GEN', 'GAN_DISC']
        self.loss_GAN_GEN = 0
        self.loss_GAN_DISC = 0
        self.loss_GAN = 0
        self.real_label = 1.
        self.fake_label = 0.
        self.loss_weights = dict()
        self.fixed_noise = torch.randn(64, 100).to(self.device)

        if self.isTrain:
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']

        self.netG = networks.define_G(ch=1, mnist=True).to(self.device)

        if self.isTrain:
            self.netD = networks.define_D(ch=1, mnist=True).to(self.device)

        if self.isTrain:
            self.fake_imgs = ImagePool(opt.pool_size)
            self.criterionGAN = nn.BCELoss()

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    # Latent input
    def set_input(self, input):
        self.noise = input.to(self.device)

    # Image input
    def set_image_input(self, real):
        self.real = real.to(self.device)

    # forward to be performed also at test time
    def forward(self):
        self.fake = self.netG(self.noise.to(self.device))

    def backward_D(self, netD, real, fake):
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, torch.ones(pred_real.shape[0]).to(self.device) * self.real_label)

        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, torch.ones(pred_fake.shape[0]).to(self.device) * self.fake_label)

        self.loss_GAN_DISC = (loss_D_real + loss_D_fake) * 0.5
        self.loss_GAN += self.loss_GAN_DISC
        self.loss_GAN_DISC.backward()

    def backward_G(self):
        self.loss_GAN_GEN = self.criterionGAN(self.netD(self.fake), torch.ones(self.fake.shape[0]).to(self.device) * self.real_label)
        self.loss_GAN = self.loss_GAN_GEN
        self.loss_GAN_GEN.backward()

    def get_images(self):
        with torch.no_grad():
            imgs = self.netG(self.fixed_noise)
        return imgs

    def update_weights(self):
        pass

    def optimize_parameters(self):
        # forward
        self.forward()
        # Train generator
        self.set_requires_grad([self.netD], False)  # Setting discriminator to no grad
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # Train discriminator
        self.set_requires_grad([self.netD], True)
        self.optimizer_D.zero_grad()
        self.backward_D(self.netD, self.real, self.fake)
        self.optimizer_D.step()
