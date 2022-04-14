import torch

_ = torch.manual_seed(123)
from torchmetrics.image.fid import FrechetInceptionDistance


def get_fid_distance(imgs_dist1, imgs_dist2):
    fid = FrechetInceptionDistance(feature=64)
    fid.update(imgs_dist1, real=True)
    fid.update(imgs_dist2, real=False)
    result = fid.compute()
    print("FID score: ", result.item())
    return result


if __name__ == "__main__":
    imgs1 = torch.randint(0, 200, (100, 3, 299, 299), dtype=torch.uint8)
    imgs2 = torch.randint(100, 255, (100, 3, 299, 299), dtype=torch.uint8)
    get_fid_distance(imgs1, imgs2)
