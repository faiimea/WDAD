import torch
from torch.utils.data import DataLoader
import cv2
import numpy as np
from torch.utils.data import Dataset,DataLoader
from PIL import Image
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

def fast_glcm(img, vmin=0, vmax=255, nbit=8, kernel_size=5):
    mi, ma = vmin, vmax
    ks = kernel_size
    h,w = img.shape
    # digitize
    bins = np.linspace(mi, ma+1, nbit+1)    # 将原来的255级别调整到8级别
    gl1 = np.digitize(img, bins) - 1    # img原有数值在bins中的位置，比如0-32就是1,32-64就是2。
    gl2 = np.append(gl1[:,1:], gl1[:,-1:], axis=1)
    # 需要被添加的数组除去第一列的gl1,添加的数组是gl1的除去最后一列，添加是沿着列添加的。
    # make glcm
    glcm = np.zeros((nbit, nbit, h, w), dtype=np.uint8)
    for i in range(nbit):
        for j in range(nbit):
            mask = ((gl1==i) & (gl2==j))
            glcm[i,j, mask] = 1
    kernel = np.ones((ks, ks), dtype=np.uint8)
    for i in range(nbit):
        for j in range(nbit):
            glcm[i,j] = cv2.filter2D(glcm[i,j], -1, kernel)
    glcm = glcm.astype(np.float32)
    return glcm

# 利用对抗前后，灰度特征变化最明显的asm参数，计算其矩阵
def fast_glcm_ASM(img, vmin=0, vmax=255, nbit=8, ks=5):
    '''
    calc glcm asm, energy
    '''
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    asm = np.zeros((h,w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            asm += glcm[i,j]**2
    ene = np.sqrt(asm)
    return asm, ene


import numpy as np


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x)
    return e_x / e_x.sum()



def main(path):
    imgs=[]
    labels=[]
    img = cv2.imread(path)
    # img = np.array(Image.open(path).convert('L'))
    # h, w = img.shape
    # asm, ene = fast_glcm_ASM(img)
    # save_path = "./grey_out.jpg"
    # cv2.imwrite(save_path, asm)
    # img = cv2.imread(save_path)
    img = torch.tensor(img)
    img = img.permute(2, 1, 0)
    img = np.array(img)
    imgs.append(img)
    labels.append(1)
    ui_dataset=MyDataset(imgs,labels)
    model = torch.load("../net/fin.pth")
    ui_dataloader = torch.utils.data.DataLoader(ui_dataset, batch_size=32)
    with torch.no_grad():
        for data in ui_dataloader:
            imgs, targets = data
            outputs = model(imgs.to(torch.float32))
            print(outputs[0])
            outputs=softmax(outputs)
            print(outputs[0])
            if(outputs.argmax(1)==0):
                temp=outputs[0][0]
                temp=float(temp*100)
                return "检测为正常图片，有"+str(temp)+"%的概率不是对抗样本"
            else:
                temp = outputs[0][1]
                temp = float(temp * 100)
                return "检测为对抗样本，有"+str(temp)+"%的概率是对抗样本"

            return outputs
    return 0