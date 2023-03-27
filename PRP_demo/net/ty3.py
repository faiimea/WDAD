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



# 定义一些超参数，如批次大小、学习速率、训练时的迭代次数等
batch_size = 32
learning_rate = 0.001
num_epochs = 10

# 定义数据增强和预处理的transforms。
# 我们使用了随机水平翻转、随机垂直翻转和随机裁剪等transforms，以增加数据集的多样性。
train_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])




# resnet网络
# So poor ...
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.resnet = nn.Sequential(*list(torchvision.models.resnet50(pretrained=True).children())[:-1])
        self.fc = nn.Linear(2048, 2)

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def predict(self, x):
        logits = self.forward(x)
        return F.softmax(logits, dim=1)




# 加载数据集。
# 我们使用ImageNet数据集中的训练集和验证集，每个图像大小为224x224。
# 构建Dataset数据集
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




train_rate=0.8


# # 在这里更改路径，以增加更多图片用来训练

# ## 数据集功能
# GRAY_img GRAY_FGSM_img 原灰度图像
# 
# new_gray_img new_gray_fgsm_img 更改过顺序的全部灰度图像
# 
# new_small_img new_small_fgsm_img 更改过顺序的小灰度图像集 是new系列的子集
# 
# TEST_img TEST_atk_img 自己测试用集
# 
# *关于PGD等图像位置，参见桌面readme.txt



# 封装成DataLoader对象
org_img_path='D:/PRP_lfz/PRP/small_gray_img'
imgPdir=os.listdir(org_img_path)
x=[]
x_t=[]
for i in range((len(imgPdir))):
    imgdir=org_img_path+"/"+imgPdir[i]
    img=cv2.imread(imgdir)
    img=torch.tensor(img)
    img=img.permute(2,1,0)
    img=np.array(img)
    if(i<len(imgPdir)*train_rate):
        x.append(img)
    else:
        x_t.append(img)
    
y=[]
y_t=[]
for i in range(len(imgPdir)):
    if(i<len(imgPdir)*train_rate):
        y.append(0)
    else:
        y_t.append(0)
#dataset=MyDataset(x,y)




# 封装成DataLoader对象
atk_img_path='D:/PRP_lfz/PRP/small_gray_fgsm_img'
imgPdir=os.listdir(atk_img_path)
for i in range(len(imgPdir)):
    imgdir=atk_img_path+"/"+imgPdir[i]
    img=cv2.imread(imgdir)
    img=torch.tensor(img)
    img=img.permute(2,1,0)
    img=np.array(img)
    if(i<len(imgPdir)*train_rate):
        x.append(img)
    else:
        x_t.append(img)

for i in range(len(imgPdir)):
    if(i<len(imgPdir)*train_rate):
        y.append(1)
    else:
        y_t.append(1)




tmp=[]
for i in range(len(x)):
    tmp.append([x[i],y[i]])
import random
random.shuffle(tmp)
x=[]
y=[]
for i in range(len(tmp)):
    x.append(tmp[i][0])
    y.append(tmp[i][1])

tmp=[]
for i in range(len(x_t)):
    tmp.append([x_t[i],y_t[i]])
import random
random.shuffle(tmp)
x_t=[]
y_t=[]
for i in range(len(tmp)):
    x_t.append(tmp[i][0])
    y_t.append(tmp[i][1])




dataset=MyDataset(x,y)
test_dataset=MyDataset(x_t,y_t)




print(len(x))
print(len(y))




train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)




# print(labels)
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

# change epochs ？
epochs = 5
steps = 0
running_loss = 0
print_every = 60




total_train_step=0
total_test_step=0




for i in range(epochs):
     print("第{}轮训练开始".format(i+1))

     for data in train_dataloader:
          imgs,targets=data
          outputs=net(imgs.to(torch.float32))
          loss=criterion(outputs,targets)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          total_train_step+=1
          if total_train_step%20==0:
               print("train_time={},loss={}".format(total_train_step, loss))
               #writer.add_scalar("train_loss",loss.item(),total_train_step)
          total_test_loss=0
     # acc
     total_accuracy=0
     with torch.no_grad():
          for data in test_dataloader:
               imgs,targets=data
               outputs=net(imgs.to(torch.float32))
               loss=criterion(outputs,targets)
               total_test_loss+=loss
               accuracy=(outputs.argmax(1)==targets).sum()
               total_accuracy=total_accuracy+accuracy

     print("total_test_loss={}".format(total_test_loss),total_test_step)
     print("total_accuracy={}".format(total_accuracy/len(test_dataset)))
     total_test_step+=1

     print("model has been saved")


# # 一个简单的测试，与训练过程无关



org_img_path='D:/PRP_lfz/PRP/TEST_img'
imgPdir=os.listdir(org_img_path)
x_t=[]
y_t=[]
for i in range((len(imgPdir))):
    imgdir=org_img_path+"/"+imgPdir[i]
    img=cv2.imread(imgdir)
    img=torch.tensor(img)
    img=img.permute(2,1,0)
    img=np.array(img)
    x_t.append(img)
    y_t.append(0)

org_img_path='D:/PRP_lfz/PRP/TEST_atk_img'
imgPdir=os.listdir(org_img_path)
for i in range((len(imgPdir))):
    imgdir=org_img_path+"/"+imgPdir[i]
    img=cv2.imread(imgdir)
    img=torch.tensor(img)
    img=img.permute(2,1,0)
    img=np.array(img)
    x_t.append(img)
    y_t.append(1)


tmp=[]
for i in range(len(x_t)):
    tmp.append([x_t[i],y_t[i]])
import random
random.shuffle(tmp)
x_t=[]
y_t=[]
for i in range(len(tmp)):
    x_t.append(tmp[i][0])
    y_t.append(tmp[i][1])




test_dataset=MyDataset(x_t,y_t)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)




total_accuracy=0
with torch.no_grad():
    for data in test_dataloader:
                imgs,targets=data
                outputs=net(imgs.to(torch.float32))
                #print(outputs.argmax(1))
                accuracy=(outputs.argmax(1)==targets).sum()
                total_accuracy+=accuracy
print(total_accuracy/len(test_dataset))




print(len(test_dataset))




print(accuracy)




print(total_accuracy/len(test_dataset))


# # 在这里保存神经网络，最好每次改名



torch.save(net,'./fin.pth')




# # 封装成DataLoader对象
# org_img_path='D:/PRP_lfz/PRP/small_gray_img'
# imgPdir=os.listdir(org_img_path)
# x=[]
# x_t=[]
# for i in range((len(imgPdir))):
#     imgdir=org_img_path+"/"+imgPdir[i]
#     img=cv2.imread(imgdir)
#     img=torch.tensor(img)
#     img=img.permute(2,1,0)
#     img=np.array(img)
#     x.append(img)
    
# y=[]
# y_t=[]
# for i in range(len(imgPdir)):
#         y.append(0)

# #dataset=MyDataset(x,y)
# # 封装成DataLoader对象
# atk_img_path='D:/PRP_lfz/PRP/small_gray_fgsm_img'
# imgPdir=os.listdir(atk_img_path)
# for i in range(len(imgPdir)):
#     imgdir=atk_img_path+"/"+imgPdir[i]
#     img=cv2.imread(imgdir)
#     img=torch.tensor(img)
#     img=img.permute(2,1,0)
#     img=np.array(img)
#     x.append(img)

# for i in range(len(imgPdir)):
#         y.append(1)




# test_dataset=MyDataset(x,y)
# test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)




# total_accuracy=0
# with torch.no_grad():
#     for data in test_dataloader:
#                 imgs,targets=data
#                 outputs=net(imgs.to(torch.float32))
#                 #print(outputs.argmax(1))
#                 accuracy=(outputs.argmax(1)==targets).sum()
#                 total_accuracy+=accuracy

# print(total_accuracy/len(test_dataset))

