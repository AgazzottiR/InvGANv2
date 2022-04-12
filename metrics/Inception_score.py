"""
from math import floor

import matplotlib as plt
import sklearn.metrics
import torch.nn.parallel
from torch.nn.functional import one_hot
import torch.utils.data
import torchvision.utils as vutils
from numpy import exp
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import std
from sklearn.cluster import KMeans

from initialization import *
from model_class_conditional_v2 import Discriminator, Generator


# assumes images have any shape and pixels in [0,255]
def calculate_inception_score(images_to_transform, label, n_split=10, eps=1E-16):
    img_size = 299
    transform_inception_v3 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    images = torch.zeros((images_to_transform.shape[0],3, img_size,img_size))
    for i in range(images.shape[0]):
        images[i] = transform_inception_v3(images_to_transform[i])
    # load cifar10 images
    print('loaded', images.shape)
    # load inception v3 model
    model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
    model.eval()
    # enumerate splits of images/predictions

    scores = list()
    n_part = floor(images.shape[0] / n_split)
    with torch.no_grad():
        for i in range(n_split):
            # retrieve images
            ix_start, ix_end = i * n_part, (i + 1) * n_part
            subset = images[ix_start:ix_end]
            label_ss = label[ix_start:ix_end]
            for l in label_ss.unique():
                print(f"label {l} is present {(label_ss == l).sum()}")
            # convert from uint8 to float32
            subset = subset.to(torch.float32)
            # scale images to the required size
            # pre-process images, scale to [-1,1]
            # predict p(y|x)
            p_yx = model(subset)
            activation = torch.nn.Softmax(dim=1)
            p_yx = activation(p_yx)
            # calculate p(y)
            p_yx = p_yx.numpy()
            p_y = expand_dims(p_yx.mean(axis=0), 0)
            # calculate KL divergence using log probabilities
            kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
            # sum over classes
            sum_kl_d = kl_d.sum(axis=1)
            # average over images
            avg_kl_d = mean(sum_kl_d)
            # undo the log
            is_score = exp(avg_kl_d)
            # store
            scores.append(is_score)
            print(f"Iteration score {i}, {is_score}")
    # average across images
    is_avg, is_std = mean(scores), std(scores)
    print('score', is_avg, is_std)
    return is_avg, is_std
"""