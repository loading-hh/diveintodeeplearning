import torch
import torchvision
from torch import nn
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt

# class FashionMNIST(object):
#     def __init__(self, **kwargs):
#         super().__init__()
#         if choice_net != "LeNet":
#             self.trans = transforms.Compose([transforms.ToTensor(), transforms.Resize([224, 224])])
#         else:
#             self.trans = transforms.ToTensor()
#
#     def load_datasets(self):
#         mnist_train = torchvision.datasets.FashionMNIST(root="C:/Users/CCU6/Practice/pytorch/data", train=True,
#                                                         transform = self.trans, download=True)
#         mnist_test = torchvision.datasets.FashionMNIST(root="C:/Users/CCU6/Practice/pytorch/data", train=False,
#                                                        transform = self.trans, download=True)
#         return mnist_train, mnist_test


class Load_my_datasets(object):
    def __init__(self, train_path, val_path):
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Imagenet中的图片的均值与方差。
        self.trans = transforms.Compose([transforms.CenterCrop((224, 224)),
                                         transforms.RandomHorizontalFlip(p=0.5),
                                         transforms.RandomVerticalFlip(p=0.5),
                                         transforms.ToTensor(),  # transforms.ToTensor()会归一化
                                         normalize])
        self.train_path = train_path
        self.val_path = val_path

    def load_datasets(self):
        train_dataset = torchvision.datasets.ImageFolder(root=self.train_path, transform=self.trans)
        val_dataset = torchvision.datasets.ImageFolder(root=self.val_path, transform=self.trans)

        return train_dataset, val_dataset


if __name__ == "__main__":
    dataset = Load_my_datasets("C:/Users/CCU6/Desktop/肺癌检测/cancer/trian", "C:/Users/CCU6/Desktop/肺癌检测/cancer/validation")
    train_dataset, val_dataset = dataset.load_datasets()
    print(train_dataset[5000])
    print(len(val_dataset))
