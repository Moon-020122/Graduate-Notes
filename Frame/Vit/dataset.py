import pandas as pd
import random 
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

class MNISTTrainDataset(Dataset):
    def __init__(self, images, labels, indicies):
        self.images = images
        self.labels = labels
        self.indicies = indicies
        self.transform = transforms.Compose([ #图片增强
            transforms.ToPILImage(), #转换为PIL图像
            transforms.RandomRotation(degrees = 15), #随机旋转
            transforms.ToTensor(), #转换为张量
            transforms.Normalize([0.5], [0.5]) #归一化
        ])
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx].reshape((28, 28)).astype(np.uint8) #将图像reshape为28*28
        label = self.labels[idx]
        index = self.indicies[idx]
        image = self.transform(image)

        return {'image': image, 'label': label, 'index': index}
    

class MNISTValDataset(Dataset):
    def __init__(self, images, labels, indicies):
        self.images = images
        self.labels = labels
        self.indicies = indicies
        self.transform = transforms.Compose([ #图片增强
            transforms.ToTensor(), #转换为张量
            transforms.Normalize([0.5], [0.5]) #归一化
        ])
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx].reshape((28, 28)).astype(np.uint8) #将图像reshape为28*28
        label = self.labels[idx]
        index = self.indicies[idx]
        image = self.transform(image)

        return {'image': image, 'label': label, 'index': index}
    

class MNISTSubmissionDataset(Dataset):
    def __init__(self, images, indicies):
        self.images = images
        self.indicies = indicies
        self.transform = transforms.Compose([ #图片增强
            transforms.ToTensor(), #转换为张量
            transforms.Normalize([0.5], [0.5]) #归一化
        ])
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx].reshape((28, 28)).astype(np.uint8) #将图像reshape为28*28
        index = self.indicies[idx]
        image = self.transform(image)

        return {'image': image, 'index': index}