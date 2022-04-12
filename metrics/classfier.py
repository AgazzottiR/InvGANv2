import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import dataset.mnist
from metrics import get_model_to_evaluate
import torchvision.utils as vutils
import numpy as np

class MnistClassifier(nn.Module):
    def __init__(self):
        super(MnistClassifier, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1,32,3,stride=2,padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(2048,100),
            nn.ReLU(),
            nn.Linear(100,10),
            nn.Softmax(dim=1)
        )

    def forward(self,x):
       return self.main(x)


if __name__ == "__main__":
    mode = "test"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if mode == "train":
        trainloader = dataset.mnist.get_mnist_trainloader()
        model = MnistClassifier()
        model.to(device)
        optim = torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))
        criterion = nn.CrossEntropyLoss()
        epochs = 100

        for n in range(epochs):
            print(f"Epoch {n}/{epochs}")
            lv = 0
            for data in trainloader:
                img, label = [d.to(device) for d in data]
                label = nn.functional.one_hot(label, num_classes=10)
                label = label.to(torch.float32)
                optim.zero_grad()
                out = model(img)
                loss = criterion(label,out)
                loss.backward()
                optim.step()
                lv += loss.detach()
            lv /= len(trainloader)
            print(f"Loss is {lv}")
        print("=> Saving parameters...")
        torch.save(model.state_dict(), r'mnist_classifier.pth')
        print("Done")
    else:
        model = MnistClassifier()
        model.load_state_dict(torch.load(r'mnist_classifier.pth'))
        testloader = dataset.mnist.get_mnist_testloader()
        img, label = next(iter(testloader))
        plt.imshow(np.transpose(vutils.make_grid(img), (1, 2, 0)))
        plt.show()
        model.eval()
        with torch.no_grad():
            out = model(img)
            out = torch.argmax(out, dim=1)
            accuracy = torch.where(out == label, 1, 0).sum() / label.shape[0]
            print(f"Accuracy on data {accuracy}")
            netMN, netG, netD = get_model_to_evaluate(dataset_name="MNIST")
            out_w, _ = netD(img.to(device))
            img_rec = netG(out_w[:,:,None,None])
            plt.imshow(np.transpose(vutils.make_grid(img_rec.cpu()),(1,2,0)))
            plt.show()
            out = model(img_rec.cpu())
            out = torch.argmax(out, dim=1)
            accuracy = torch.where(out == label, 1, 0).sum() / label.shape[0]
            print(f"Accuracy on reconstructed data {accuracy}")


