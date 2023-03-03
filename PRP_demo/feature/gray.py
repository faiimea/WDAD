import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
def fast_glcm(img, vmin=0, vmax=255, nbit=8, kernel_size=5):
    mi, ma = vmin, vmax
    ks = kernel_size
    h,w = img.shape
    # digitize
    bins = np.linspace(mi, ma+1, nbit+1)#将原来的255级别调整到8级别
    gl1 = np.digitize(img, bins) - 1#img原有数值在bins中的位置，比如0-32就是1,32-64就是2。
    gl2 = np.append(gl1[:,1:], gl1[:,-1:], axis=1)
    #需要被添加的数组除去第一列的gl1,添加的数组是gl1的除去最后一列，添加是沿着列添加的。
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

def fast_glcm_mean(img, vmin=0, vmax=255, nbit=8, ks=5):
    '''
    calc glcm mean
    '''
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    mean = np.zeros((h,w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            mean += glcm[i,j] * i / (nbit)**2
    return mean

def fast_glcm_std(img, vmin=0, vmax=255, nbit=8, ks=5):
    '''
    calc glcm std
    '''
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    mean = np.zeros((h,w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            mean += glcm[i,j] * i / (nbit)**2
    std2 = np.zeros((h,w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            std2 += (glcm[i,j] * i - mean)**2
    std = np.sqrt(std2)
    return std

def fast_glcm_contrast(img, vmin=0, vmax=255, nbit=8, ks=5):
    '''
    calc glcm contrast
    '''
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    cont = np.zeros((h,w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            cont += glcm[i,j] * (i-j)**2
    return cont

def fast_glcm_dissimilarity(img, vmin=0, vmax=255, nbit=8, ks=5):
    '''
    calc glcm dissimilarity
    '''
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    diss = np.zeros((h,w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            diss += glcm[i,j] * np.abs(i-j)
    return diss

def fast_glcm_homogeneity(img, vmin=0, vmax=255, nbit=8, ks=5):
    '''
    calc glcm homogeneity
    '''
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    homo = np.zeros((h,w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            homo += glcm[i,j] / (1.+(i-j)**2)
    return homo

def fast_glcm_ASM(img, vmin=0, vmax=255, nbit=8, ks=5):
    '''
    calc glcm asm, energy
    '''
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    asm = np.zeros((h,w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            asm  += glcm[i,j]**2
    ene = np.sqrt(asm)
    return asm, ene

def fast_glcm_max(img, vmin=0, vmax=255, nbit=8, ks=5):
    '''
    calc glcm max
    '''
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    max_  = np.max(glcm, axis=(0,1))
    return max_

def fast_glcm_entropy(img, vmin=0, vmax=255, nbit=8, ks=5):
    '''
    calc glcm entropy
    '''
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    pnorm = glcm / np.sum(glcm, axis=(0,1)) + 1./ks**2
    ent  = np.sum(-pnorm * np.log(pnorm), axis=(0,1))
    return ent

#应用于本地图片
image = r'PRP/dl_learn/output1_1.png'
img=np.array(Image.open(image).convert('L'))
#img = data.camera()
plt.imshow(img,cmap=plt.cm.gray)
h,w = img.shape

#计算各个统计值
mean = fast_glcm_mean(img)
std = fast_glcm_std(img)
cont = fast_glcm_contrast(img)
diss = fast_glcm_dissimilarity(img)
homo = fast_glcm_homogeneity(img)
asm, ene = fast_glcm_ASM(img)
ma = fast_glcm_max(img)
ent = fast_glcm_entropy(img)
#绘制各个特征的子图
plt.figure(figsize=(10,4.5))
fs = 15
plt.subplot(2,5,1)
plt.tick_params(labelbottom=False, labelleft=False)
plt.imshow(img,cmap=plt.cm.gray)
plt.title('original', fontsize=fs)
plt.subplot(2,5,2)
plt.tick_params(labelbottom=False, labelleft=False)
plt.imshow(mean)
plt.title('mean', fontsize=fs)
plt.subplot(2,5,3)
plt.tick_params(labelbottom=False, labelleft=False)
plt.imshow(std)
plt.title('std', fontsize=fs)
plt.subplot(2,5,4)
plt.tick_params(labelbottom=False, labelleft=False)
plt.imshow(cont)
plt.title('contrast', fontsize=fs)

plt.subplot(2,5,5)
plt.tick_params(labelbottom=False, labelleft=False)
plt.imshow(diss)
plt.title('dissimilarity', fontsize=fs)
plt.subplot(2,5,6)
plt.tick_params(labelbottom=False, labelleft=False)
plt.imshow(homo)
plt.title('homogeneity', fontsize=fs)
plt.subplot(2,5,7)
plt.tick_params(labelbottom=False, labelleft=False)
plt.imshow(asm)
plt.title('ASM', fontsize=fs)
plt.subplot(2,5,8)
plt.tick_params(labelbottom=False, labelleft=False)
plt.imshow(ene)
plt.title('energy', fontsize=fs)
plt.subplot(2,5,9)
plt.tick_params(labelbottom=False, labelleft=False)
plt.imshow(ma)
plt.title('max', fontsize=fs)
plt.subplot(2,5,10)
plt.tick_params(labelbottom=False, labelleft=False)
plt.imshow(ent)
plt.title('entropy', fontsize=fs)
plt.tight_layout(pad=0.5)

#plt.show()
print(img.shape)