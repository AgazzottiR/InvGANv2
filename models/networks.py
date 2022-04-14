import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('Normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, lr_policy, opt):
    if lr_policy == 'linear':
        def lamda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lamda_rule)
    elif lr_policy == 'step':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[n for n in range(15, opt.n_epochs, 30)], gamma=1)
    elif lr_policy == 'step_grow':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[n for n in range(15, opt.n_epochs, 30)], gamma=1)
    elif lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplemented('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert torch.cuda.is_available()
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, init_gain=init_gain)
    return net


# input_nc, output_nc, nfg, netG='vanilla', norm='batch', use_dropout=False, init_type='normal', init_gain=0.02,
#              gpu_ids=[]
def define_G(ch,mnist=False):
    if mnist:
        return GeneratorMnist(output_nc=ch)
    return Generator(output_nc=ch)


# input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=.02, gpu_ids=[]
def define_D(ch,mnist=False):
    if mnist:
        return DiscriminatorMnist(ic=ch)
    return Discriminator(ic=ch)


def define_M():
    return MappingNetwork()


class MappingNetwork(nn.Module):
    def __init__(self):
        super(MappingNetwork, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 100),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)[:, :, None, None]


class Generator(nn.Module):
    def __init__(self, input_nc=100, output_nc=3):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(input_nc, 512, 4, 2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, output_nc, 4, 2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


class GeneratorMnist(nn.Module):
    def __init__(self, input_nc=100, output_nc=1):
        super(GeneratorMnist, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(100, 7 * 7 * 256),
            nn.BatchNorm1d(7 * 7 * 256),
            nn.ReLU())
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 5, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 5, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 5, 1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, output_nc, 5, 2, padding=3, output_padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.conv(self.linear(x).reshape(x.shape[0], 256, 7, 7))

class DiscriminatorMnist(nn.Module):
    def __init__(self, ic=1):
        super(DiscriminatorMnist, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(ic, 32, 5, 1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 5, 2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 5, 2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, 5, 3, padding=1),
            nn.Flatten(start_dim=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.main(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, ic=3):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(ic, 32, 5, 2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, 2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 5, 2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 101, 5, 2, padding=1),
        )
        self.linear = nn.Sequential(
            nn.Linear(256, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, 1)
        )
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        x = self.main(x)
        x = x.reshape((x.shape[0], -1))
        out_z, out_rf = x[:, 0:100], self.sigm(x[:, -1])
        return out_z, out_rf

    def forward_feature(self, x, linear=False):
        fm = [x]
        fm_names = ['input']
        for layer in self.main.children():
            fm.append(layer(fm[-1]))
            fm_names.append(str(layer))
        fm[-1] = fm[-1].reshape((fm[-1].shape[0], -1))
        if linear:
            for layer in self.linear.children():
                fm.append(layer(fm[-1]))
                fm_names.append(str(layer))
        fm.append(fm[-1][:, 0:100])
        fm_names.append("Z_output")
        fm.append(self.sigm(fm[-1][:, -1]))
        fm_names.append("BCE_output")
        return fm, fm_names


class MMDLoss(nn.Module):
    def __init__(self):
        super(MMDLoss, self).__init__()

    def forward(self, x, y, kernel='rbf'):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
        rx = (xx.diag().unsqueeze(0).expand_as(xx))
        ry = (yy.diag().unsqueeze(0).expand_as(yy))

        dxx = rx.t() + rx - 2. * xx  # Used for A in (1)
        dyy = ry.t() + ry - 2. * yy  # Used for B in (1)
        dxy = rx.t() + ry - 2. * zz  # Used for C in (1)

        XX, YY, XY = (torch.zeros(xx.shape).to(device),
                      torch.zeros(xx.shape).to(device),
                      torch.zeros(xx.shape).to(device))

        if kernel == "multiscale":

            bandwidth_range = [0.2, 0.5, 0.9, 1.3]
            for a in bandwidth_range:
                XX += a ** 2 * (a ** 2 + dxx) ** -1
                YY += a ** 2 * (a ** 2 + dyy) ** -1
                XY += a ** 2 * (a ** 2 + dxy) ** -1

        if kernel == "rbf":

            bandwidth_range = [10, 15, 20, 50]
            for a in bandwidth_range:
                XX += torch.exp(-0.5 * dxx / a)
                YY += torch.exp(-0.5 * dyy / a)
                XY += torch.exp(-0.5 * dxy / a)

        return torch.mean(XX + YY - 2. * XY)
