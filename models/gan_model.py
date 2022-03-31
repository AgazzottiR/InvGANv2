import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import torch.nn as nn


class GanModel(BaseModel):
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_name = ['GAN']
        self.loss_GAN = 0
        visual_names_A = ['real', 'fake']
        self.real_label = 1.
        self.fake_label = 0.

        if self.isTrain:
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']

        self.netG = networks.define_G().to(self.device)

        if self.isTrain:
            self.netD = networks.define_D().to(self.device)

        if self.isTrain:
            self.fake_imgs = ImagePool(opt.pool_size)
            self.criterionGAN = nn.BCELoss()

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    # Latent input
    def set_input(self, input):
        # Evenutual input preprocessing
        if len(input.shape) == 2:
            input = input[:, :, None, None]
        elif len(input.shape) < 2:
            raise Exception('Input shape must be [bach_size, latent_size]')
        assert input.shape[1] == self.opt.latent_size
        self.noise = input.to(self.device)

    # Image input
    def set_image_input(self, real):
        self.real = real.to(self.device)

    # forward to be performed also at test time
    def forward(self):
        self.fake = self.netG(self.noise.to(self.device))

    def backward_D(self, netD, real, fake):
        _, pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, torch.ones(pred_real.shape[0]).to(self.device)*self.real_label)

        _, pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, torch.ones(pred_fake.shape[0]).to(self.device)*self.fake_label)

        loss_D = (loss_D_real + loss_D_fake) * 0.5
        self.loss_GAN += loss_D
        loss_D.backward()
        return loss_D

    def backward_G(self):
        lossG_gan = self.criterionGAN(self.netD(self.fake)[1],torch.ones(self.fake.shape[0]).to(self.device)*self.real_label)
        self.lossG = lossG_gan
        self.loss_GAN = self.lossG
        self.lossG.backward()

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
