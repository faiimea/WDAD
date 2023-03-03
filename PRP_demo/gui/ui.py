import tkinter as tk
from tkinter.filedialog import askopenfilename
from PIL import Image,ImageTk
import torch.nn as nn
import torchvision
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import main
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

class Test():
    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry('1000x750+150+100')
        self.root.title('detection')
        self.text = tk.StringVar()
        self.path=tk.StringVar()
        self.text.set("Detection")
        self.label = tk.Label(self.root, textvariable=self.text)

        self.button = tk.Button(self.root,
                                text="Click to 选择图片",
                                command=self.choosepic)
        self.button.pack()
        self.label.pack()
        self.e1 = tk.Entry(self.root, state='readonly', text=self.path, width=50)
        self.e1.pack()
        self.l1 = tk.Label(self.root)
        self.l1.pack()
        self.root.mainloop()


    def choosepic(self):
        global path_
        global flag
        flag = 1
        path_ = askopenfilename()
        self.path.set(path_)
        img_open = Image.open(self.e1.get())
        print(self.e1.get())
        img = ImageTk.PhotoImage(img_open)
        self.l1.config(image=img)
        self.l1.image = img  # keep a reference
        self.root.update()
        # self.text.set(test.just_test(self.e1.get())) Correct
        self.text.set(main.main(self.e1.get()))


app=Test()