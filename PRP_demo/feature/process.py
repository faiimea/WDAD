import os
import cv2
import numpy as np
from PIL import Image

# 对图像进行灰度处理，得到灰度共生矩阵
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

# 处理单张图片

img_path="./output.png"
img=np.array(Image.open(img_path).convert('L'))
h,w = img.shape
asm,ene=fast_glcm_ASM(img)
#asm即为所需
save_path="./grey_out.jpg"
cv2.imwrite(save_path,asm)

# 批量灰度处理图片：

# imgPdir=os.listdir(img_path)
# for i in range(len(imgPdir)):
#     imgdir=img_path+"/"+imgPdir[i]
#     img=np.array(Image.open(imgdir).convert('L'))
#     h,w = img.shape
#     asm,ene=fast_glcm_ASM(img)
#     #asm即为所需
#     save_path="D:/PRP_lfz/PRP/GRAY_FGSM_img/"+str(i)+".jpg"
#     cv2.imwrite(save_path,asm)
