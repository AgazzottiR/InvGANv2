import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import torch.nn as nn


class InvGanModel(BaseModel):
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_name = ['GAN','GAN_D','GAN_G', 'L2_DISC', 'L2_GEN', 'MMD', 'FL', 'L2_MN', 'MMD_MN']
        self.loss_GAN_D = 0
        self.loss_GAN_G = 0
        self.loss_GAN = 0
        self.loss_L2_DISC = 0
        self.loss_L2_GEN = 0
        self.loss_MMD = 0
        self.loss_FL = 0
        self.loss_L2_MN = 0
        self.loss_MMD_MN = 0
        self.previous_batch = None

        self.real_label = 1.
        self.fake_label = 0.

        if self.isTrain:
            self.model_names = ['G', 'D', 'M']
        else:
            self.model_names = ['G', 'M']

        self.netG = networks.define_G().to(self.device)
        self.netM = networks.define_M().to(self.device)

        if self.isTrain:
            self.netD = networks.define_D().to(self.device)

        if self.isTrain:
            self.fake_imgs = ImagePool(opt.pool_size)
            self.criterionGAN = nn.BCELoss()
            self.criterionL2 = nn.MSELoss()
            self.criterionMMD = networks.MMDLoss()

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_M = torch.optim.Adam(self.netM.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_M)

    # Latent input
    def set_input(self, input):
        # Evenutual input preprocessing
        self.noise = input.to(self.device)

    # Image input
    def set_image_input(self, real):
        self.real = real.to(self.device)


    def forward(self):
        self.w = self.netM(self.noise.to(self.device))
        if self.isTrain:
            self.w_no_grad = self.w.detach()
            self.compute_previous_batch()
            self.fake = self.netG(self.w_no_grad)
        else:
            self.fake = self.netG(self.w)

    def compute_previous_batch(self):
        try:
            if self.previous_batch is not None:
                torch.cat((self.previous_batch, self.output_w_real.detach()), dim=0)
            else:
                self.previous_batch = self.output_w_real.detach()
            self.previous_batch = self.previous_batch[torch.randperm(self.previous_batch.shape[0])]
            self.fake = torch.cat((self.w_no_grad[:self.fake.shape[0] // 2],
                                   self.previous_batch[self.fake.shape[:self.fake.shape[0] // 2]]), dim=0)
        except:
            pass

    def backward_D(self):

        self.output_w_real, self.pred_real = self.netD(self.real)
        loss_D_real = self.criterionGAN(self.pred_real, torch.ones(self.pred_real.shape[0], requires_grad=True).to(
            self.device) * self.real_label)

        self.output_w_fake, self.pred_fake = self.netD(self.fake.detach())
        loss_D_fake = self.criterionGAN(self.pred_fake, torch.ones(self.pred_fake.shape[0], requires_grad=True).to(
            self.device) * self.fake_label)

        self.loss_L2_DISC = self.criterionL2(self.w_no_grad.reshape(self.w_no_grad.shape[0:2]), self.output_w_fake)
        self.loss_MMD = self.criterionMMD(self.w_no_grad.reshape(self.w_no_grad.shape[0:2]), self.output_w_real)

        self.loss_GAN_D = (loss_D_real + loss_D_fake) * 0.5 # + self.loss_MMD + self.loss_L2_DISC
        self.loss_GAN += self.loss_GAN_D
        self.loss_GAN_D.backward()

    def backward_M(self):
        self.loss_L2_MN = self.criterionL2(self.w.reshape(self.w.shape[0:2]), self.output_w_fake.detach())
        self.loss_MMD_MN = self.criterionMMD(self.w.reshape(self.w.shape[0:2]), self.output_w_real.detach())
        self.loss_MN = self.loss_MMD_MN + self.loss_L2_MN
        # self.loss_MN.backward()

    def backward_G(self):
        self.w_no_grad_after_disc = self.output_w_real.detach()
        self.fake_2 = self.netG(self.w_no_grad_after_disc[:,:,None,None])
        self.out_w_2, _ = self.netD(self.fake_2)
        lossG_gan = self.criterionGAN(self.netD(self.fake)[1], torch.ones(self.fake.shape[0], requires_grad=True).to(
            self.device) * self.real_label)
        if self.previous_batch is not None:
            self.loss_L2_GEN = self.criterionL2(self.out_w_2, self.w_no_grad_after_disc)
        self.loss_GAN_G = lossG_gan + self.loss_L2_GEN
        self.loss_GAN = self.loss_GAN_G
        self.loss_GAN_G.backward()

    def optimize_parameters(self):
        # forward
        self.forward()
        # Train discriminator
        self.set_requires_grad([self.netG, self.netM], False)
        self.set_requires_grad([self.netD], True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        # Train Mapping
        self.set_requires_grad([self.netD, self.netG], False)
        self.set_requires_grad([self.netM], True)
        self.optimizer_M.zero_grad()
        self.backward_M()
        self.optimizer_M.step()
        # Train generator
        self.set_requires_grad([self.netD, self.netM], False)  # Setting discriminator to no grad
        self.set_requires_grad([self.netG], True)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
