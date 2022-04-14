from math import floor

from torchmetrics.image.inception import InceptionScore
import torch.nn.parallel
import torch.utils.data
from numpy import exp
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import std
from torchvision.transforms import transforms

from metrics.classfier import classifyMnist
from models.networks import Generator

_ = torch.manual_seed(123)


def trasform_for_Iv3(images_to_transform, nc=3):
    img_size = 299
    transform_inception_v3 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    images = torch.zeros((images_to_transform.shape[0], nc, img_size, img_size)).to(torch.uint8)
    for i in range(images.shape[0]):
        images[i] = transform_inception_v3(images_to_transform[i])
    return images

# assumes images have any shape and pixels in [0,255]
def calculate_inception_score(images_to_transform, label, n_split=10, eps=1E-16, nc=3):
    print("Function Deprecated should call get_inception_score(img)")
    images = trasform_for_Iv3(images_to_transform)
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


def get_inception_score(imgs):
    inception = InceptionScore()
    inception.update(imgs)
    result = inception.compute()
    print("Inception Score is (avg,var): ", [a.item() for a in result])
    return result

if __name__ == "__main__":
    imgs = torch.randint(0, 255, (100, 3, 32, 32), dtype=torch.uint8)
    imgs = trasform_for_Iv3(imgs)
    get_inception_score(imgs)
