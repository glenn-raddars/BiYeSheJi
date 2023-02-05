import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image

def read_voc_images(root='dataset/VOCdevkit/VOC2012', is_train=True, max_num=None):
    txt_fname = "%s/ImageSets/Segmentation/%s" % (root, "trainval.txt" if is_train else "test.txt")
    with open(txt_fname, 'r') as f:
        images = f.read().split() # 拆分成一个一个名字组成list

    if max_num is not None:
        images = images[:min(max_num, len(images))] # 限制最大图片数量

    features, labels = [], []
    for fname in images:
        features.append(Image.open("%s/JPEGImages/%s.jpg"%(root, fname)).convert("RGB"))
        labels.append(Image.open("%s/SegmentationClass/%s.png"%(root, fname)).convert("RGB"))

    return features, labels

def show_images(imgs, num_rows, num_cols): # num_rows有几行，num_cols一行有几张图
    _, axes = plt.subplots(num_rows, num_cols)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i*num_cols+j])
    
    plt.show()
    return axes

if __name__ == "__main__":
    features, labels = read_voc_images()
    imgs = features[0:5] + labels[0:5]
    show_images(imgs, 2, 5)
