import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import torch.nn as nn


class InvGanModel(BaseModel):
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_name = ['GAN', 'GAN_D', 'GAN_GEN', 'L2_D', 'L2_GEN', 'MMD_D', 'FL_GEN', 'L2_MN', 'MMD_MN', 'GAN_MN', 'D',
                          'G']
        self.loss_GAN_D = 0
        self.loss_GAN_GEN = 0
        self.loss_GAN = 0
        self.loss_L2_D = 0
        self.loss_L2_GEN = 0
        self.loss_MMD_D = 0
        self.loss_FL_GEN = 0
        self.loss_L2_MN = 0
        self.loss_MMD_MN = 0
        self.loss_GAN_MN = 0
        self.loss_MN = 0
        self.loss_D = 0
        self.loss_G = 0
        self.previous_batch = None
        self.fixed_noise = torch.randn((64, 512)).to(self.device)

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

            self.loss_weights = dict()
            for ln in self.loss_name:
                ln = ln + '_W'
                if ln.startswith('_'):
                    continue
                self.loss_weights[ln] = 1.

    def update_weights(self,mf_d=3):
        for w in self.loss_weights:
            mf = mf_d
            if '_MN' in w or '_GEN' in w:
                if self.loss_weights[w] > 10:
                    mf = 1
                self.loss_weights[w] *= mf
                print(f"Updating loss {w.strip('_W')} multiplying by a factor {mf}: {self.loss_weights[w]/mf} --> {self.loss_weights[w]}")

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
            if self.previous_batch.shape[0] > self.output_w_real.shape[0] * 5:  # Clean previous batch
                self.previous_batch = self.previous_batch[:self.output_w_real.shape[0]]
        except:
            pass

    def backward_D(self):
        self.output_w_real, self.pred_real = self.netD(self.real)
        loss_D_real = self.criterionGAN(self.pred_real, torch.ones(self.pred_real.shape[0], requires_grad=True).to(
            self.device) * self.real_label)

        self.output_w_fake, self.pred_fake = self.netD(self.fake.detach())
        loss_D_fake = self.criterionGAN(self.pred_fake, torch.ones(self.pred_fake.shape[0], requires_grad=True).to(
            self.device) * self.fake_label)

        self.loss_L2_D = self.criterionL2(self.w_no_grad.reshape(self.w_no_grad.shape[0:2]), self.output_w_fake)  * self.loss_weights['L2_D_W']
        self.loss_MMD_D = self.criterionMMD(self.w_no_grad.reshape(self.w_no_grad.shape[0:2]), self.output_w_real) * self.loss_weights['MMD_D_W']

        self.loss_GAN_D = (loss_D_real + loss_D_fake) * 0.5 * self.loss_weights['GAN_D_W']
        self.loss_D = self.loss_GAN_D + self.loss_MMD_D + self.loss_L2_D
        self.loss_GAN = (loss_D_real + loss_D_fake) * 0.5 * self.loss_weights['GAN_D_W']
        self.loss_D.backward()

    def backward_M(self):  # cambia con anche loss generatore
        self.w = self.netM(self.noise)
        self.fake_rec = self.netG(self.w)
        self.w_rec, _ = self.netD(self.fake_rec)
        self.w_real_rec, _ = self.netD(self.real)
        self.loss_GAN_MN = self.criterionGAN(self.netD(self.fake_rec)[1],
                                             torch.ones(self.fake.shape[0], requires_grad=True).to(
                                                 self.device) * self.real_label) * self.loss_weights['GAN_MN_W']
        self.loss_L2_MN = self.criterionL2(self.w.reshape(self.w.shape[0:2]),
                                           self.w_rec.reshape(self.w_rec.shape[0:2])) * self.loss_weights['L2_MN_W']
        self.loss_MMD_MN = self.criterionMMD(self.w.reshape(self.w.shape[0:2]), self.w_real_rec) * self.loss_weights[
            'MMD_MN_W']

        self.loss_MN = self.loss_MMD_MN + self.loss_L2_MN + self.loss_GAN_MN
        self.loss_MN.backward()

    def get_images(self):
        with torch.no_grad():
            img = self.netG(self.netM(self.fixed_noise))
            img_rec = self.netG(self.netD(img)[0][:, :, None, None])
        return img, img_rec

    def backward_G(self):
        self.w_real, _ = self.netD(self.real)
        self.real_rec = self.netG(self.w_real[:, :, None, None])
        self.w_real_rec, _ = self.netD(self.real_rec)

        self.fm, _ = self.netD.forward_feature(self.real)
        self.fm_rec, _ = self.netD.forward_feature(self.netG(self.fm[-2][:, :, None, None]))

        self.loss_GAN_GEN = self.criterionGAN(self.netD(self.fake)[1],
                                            torch.ones(self.fake.shape[0], requires_grad=True).to(
                                                self.device) * self.real_label) * self.loss_weights['GAN_GEN_W']
        self.loss_FL_GEN = self.criterionL2(self.fm[-4], self.fm_rec[-4]) * self.loss_weights['FL_GEN_W']
        self.loss_L2_GEN = self.criterionL2(self.w_real_rec, self.w_real) * self.loss_weights['L2_GEN_W']

        self.loss_G = self.loss_GAN_GEN + self.loss_L2_GEN + self.loss_FL_GEN
        self.loss_GAN += self.loss_GAN_GEN
        self.loss_G.backward()

    def free_optimizers(self):
        for optim in self.optimizers:
            optim.zero_grad()
        self.netD.zero_grad()
        self.netM.zero_grad()
        self.netG.zero_grad()

    def step_all(self):
        for optim in self.optimizers:
            optim.step()

    def optimize_parameters(self):
        # forward
        self.forward()
        # Train discriminator
        self.set_requires_grad([self.netG, self.netM], False)
        self.set_requires_grad([self.netD], True)
        self.free_optimizers()
        self.backward_D()
        self.optimizer_D.step()
        # Train Mapping
        self.set_requires_grad([self.netD, self.netG], False)
        self.set_requires_grad([self.netM], True)
        self.free_optimizers()
        self.backward_M()
        self.optimizer_M.step()
        # Train generator
        self.set_requires_grad([self.netD, self.netM], False)  # Setting discriminator to no grad
        self.set_requires_grad([self.netG], True)
        self.free_optimizers()
        self.backward_G()
        self.optimizer_G.step()
