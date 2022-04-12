import sklearn.metrics
import torch.nn.parallel
import torch.utils.data
from sklearn.cluster import KMeans
from torchvision.transforms import transforms

from dataset import cifar10, mnist
from metrics import get_model_to_evaluate


def Kmeans_DCGAN_feature_extractor(dataset_name='CIFAR'):
    batch_size_kmeans = 4000
    image_size = 32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transform_km = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4821, 0.4465), (0.2467, 0.2431, 0.2611)),
    ])
    if dataset_name == 'CIFAR':
        testloader = cifar10.get_cifar10_testloader(batch_size_kmeans)
        trainloader = cifar10.get_cifar10_trainloader(batch_size_kmeans)
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    elif dataset_name == 'MNIST':
        trainloader = mnist.get_mnist_trainloader(batch_size_kmeans)
        testloader = mnist.get_mnist_testloader(batch_size_kmeans)
        classes = [str(i) for i in range(10)]
    else:
        raise NotImplemented()

    netMN, netG, netD = get_model_to_evaluate(dataset_name=dataset_name)
    # fixed_noise = torch.randn(4000, 512, 1, 1, device=device)
    # label = torch.randint(0, 10, (4000,), device=device).to(torch.int64)

    Kmeans = KMeans(n_clusters=10)
    img, label = next(iter(trainloader))
    img = img.to(device)
    fm, _ = netD.forward_feature(img)
    prediction_real = Kmeans.fit_predict(fm[-2].reshape(batch_size_kmeans, -1).detach().cpu().numpy(), label)
    # predict
    img, label = next(iter(trainloader))
    img = img.to(device)
    fm, _ = netD.forward_feature(img)
    w_rec, _ = netD(img)
    img_rec = netG(w_rec[:,:,None,None])
    fm_rec,_ = netD.forward_feature(img_rec)

    prediction_reconstruct = Kmeans.predict(fm[-2].reshape(batch_size_kmeans,-1).detach().cpu().numpy())
    accuracy_real = sklearn.metrics.accuracy_score(label,prediction_real)
    accuracy_rec = sklearn.metrics.accuracy_score(label, prediction_reconstruct)
    print(f"Accuracy {accuracy_real}, accuracy reconstruct {accuracy_rec}\n")

if __name__ == "__main__":
    Kmeans_DCGAN_feature_extractor(dataset_name="CIFAR")