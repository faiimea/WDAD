import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.nn import Sequential, Conv2d,MaxPool2d,Flatten,Linear
import torchvision
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import os

class MyDataset(Dataset):#需要继承torch.utils.data.Dataset
    def __init__(self,feature,target):
        super(MyDataset, self).__init__()
        self.feature =feature
        self.target = target
    def __getitem__(self,index):
        item=self.feature[index]
        label=self.target[index]
        return item,label
    def __len__(self):
        return len(self.feature)

def main(path):
    imgs=[]
    labels=[]
    img = cv2.imread(path)
    img = torch.tensor(img)
    img = img.permute(2, 1, 0)
    img = np.array(img)
    imgs.append(img)
    labels.append(1)
    ui_dataset=MyDataset(imgs,labels)
    model = torch.load("../net/fin.pth")
    ui_dataloader = torch.utils.data.DataLoader(ui_dataset, batch_size=1)
    with torch.no_grad():
        for data in ui_dataloader:
            imgs, targets = data
            outputs = model(imgs.to(torch.float32))
            print(outputs)
            return "org_outpus"
    return "org_outpus"