import os

import torchvision.utils
import numpy as np
from options.test_options import TestOptions
from dataset import create_dataset
from models import create_model
from util.visualize import save_images
from util import html
import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 0
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are
    # needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
    dataset = [torch.randn(64, 100, 1, 1) for x in range(100)]
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers

    # initialize logger
    if opt.use_wandb:
        wandb.login(key='bb1abfc16a616453716180cdc3306cf7ce03d891')
        wandb.init(project="InvGAN", entity="riccardoagazzotti")
        wandb.config = {
            "learning_rate": 0.0002,
            "epochs": 200,
            "batch_size": 128
        }
        wandb_run = wandb.init(project='InvGAN', name=opt.name, config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='InvGAN')

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name,
                           '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference
        visuals = model.get_current_visuals()
        plt.imshow(np.transpose(vutils.make_grid(visuals['fake']),(1,2,0)))
        plt.show()
        img_path = model.get_image_paths()
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)
    webpage.save()  # save the HTML
