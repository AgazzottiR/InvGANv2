import time

import torch
import wandb

from options.train_options import TrainOptions
from models import create_model
from util.visualize import Visualizer
from dataset import cifar10

if __name__ == "__main__":
    opt = TrainOptions().parse()
    dataloader = cifar10.get_cifar10_trainloader(batch_size=opt.batch_size)
    data_size = len(dataloader)
    print('Training images = %d' % data_size)
    t_data = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = create_model(opt)
    try:
        model.load_networks('latest')
    except:
        print("Default init for parameters")
    model.setup(opt)
    visualizer = Visualizer(opt)
    total_iters = 0

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        visualizer.reset()
        for i, data in enumerate(dataloader):
            data = [d.to(device) for d in data]
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time + iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(torch.randn((opt.batch_size, opt.latent_size), device=device))
            model.set_image_input(data[0])
            model.optimize_parameters()

            if total_iters % opt.print_freq == 0:
                img, img_rec = model.get_images()
                visualizer.plot_images(img, img_rec)
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / data_size, losses, model.loss_weights,
                                                   model.optimizers)
            if total_iters % opt.save_latest_freq == 0:
                print("Saving checkpoint")
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        model.update_learning_rate()

        if epoch % opt.save_epoch_freq == 0:
            model.save_networks('latest')
            model.save_networks(epoch)
        if (epoch + 1) % 10 == 0:
            model.update_weights()
        print('End of epoch %d / %d \t Time Taken: %d sec' % (
        epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
