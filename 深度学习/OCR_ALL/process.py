import os
import numpy as np
import cv2
from matplotlib import pyplot as plt


# 解决读取路径为中文的问题
def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), 0)
    return cv_img

# 解决保存路径有中文的问题
def cv_imwrite(savePath,tem):
    cv2.imencode('.png',tem)[1].tofile(savePath)


# 得到训练所用的标签
def get_labels(basic_path):
    # 数据集所在路径
    label_path=os.path.join(basic_path,'test\\')

    # 获取所有文件名，即为全部500个汉字，即为汉字标签
    char_labels=os.listdir(label_path)

    # 创建一个单位矩阵，每一列为一个one-hot向量，作为训练标签
    labels=np.identity(len(char_labels))
    return labels,char_labels


# 图像预处理
def pre_process(basic_path,*args):
    '''
    图像预处理，将图像去除背景以及噪声点，然后二值化，
    basic_path示例为：''E:\OCR\char''
    *args中传入'train'或者'test'，用于选择处理对应文件夹下的图像
    '''
    # 训练集或者测试集的路径
    train_path=os.path.join(basic_path,args[0])
    # 获取当前路径下所有汉字，存到chars中
    chars=os.listdir(train_path)
    for char in chars:
        print(char+'正在处理中。。。。。')
        # 创建文件夹
        os.makedirs('.\\test_process'+'\\'+args[0]+'\\'+char)
        # 每一个汉字文件夹的路径
        pic_path=os.path.join(train_path,char)
        # 获取当前汉字文件夹下所有图片的名字
        pic_names=os.listdir(pic_path)
        for name in pic_names:
            # 带有背景的图片
            if name[0]=='b':
                img=cv_imread(os.path.join(pic_path,name))
                for i in range(img.shape[0]):
                    for j in range(img.shape[1]):
                        if img[i,j]>146:
                            img[i,j]=255
                        else:
                            img[i,j]=0 
            # 带有噪声点的图片
            elif name[0]=='g':
                img=cv_imread(os.path.join(pic_path,name))
                for i in range(img.shape[0]):
                    for j in range(img.shape[1]):
                        if img[i,j]<20 or img[i,j]>250:
                            img[i,j]=255
                        else:
                            img[i,j]=0 
            # 普通图片
            else:
                img=cv_imread(os.path.join(pic_path,name))
                for i in range(img.shape[0]):
                    for j in range(img.shape[1]):
                        if img[i,j]<250:
                            img[i,j]=0
                        else:
                            img[i,j]=255
            img=cv2.resize(img,(128,128),interpolation=cv2.INTER_CUBIC)
            # print(img.shape)
            cv_imwrite(os.path.join(r'.\test_process',args[0],char,name),img)


    
if __name__=='__main__':
    pre_process(r'.\orig_dataset','train')
    
    
    