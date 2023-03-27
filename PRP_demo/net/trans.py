#!/usr/bin/env python
# coding: utf-8



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




org_img_path='D:/PRP_lfz/PRP/GRAY_FGSM_img'
imgPdir=os.listdir(org_img_path)
img_lst=[]
for i in range((len(imgPdir))):
    imgdir=org_img_path+"/"+imgPdir[i]
    img=cv2.imread(imgdir)
    img_lst.append(img)

import random
random.shuffle(img_lst)




for i in range(len(img_lst)):
    save_path="D:/PRP_lfz/PRP/new_gray_fgsm_img/"+str(i)+".jpg"
    cv2.imwrite(save_path,img_lst[i])






