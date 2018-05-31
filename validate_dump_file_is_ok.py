
import os
import numpy as np
from scipy import ndimage
import _pickle as pickle
import pandas as pd
import cv2

train_filename='train_dump_img.npy'
train_groundtruth='train_dump_groundtruth.npy'
folder='/home/hung/Desktop/model to compare/Recursive-CNNs-server_branch/resized/'
val_filename='val_dump_img.npy'
val_groundtruth='val_dump_groundtruth.npy'

train=np.load(train_filename)
train_label=np.load(train_groundtruth)
random=100

a=np.around(train[random]*255.0+128.0)
b=np.empty_like(a)
b[:,:,0]=a[:,:,2]
b[:,:,1]=a[:,:,1]
b[:,:,2]=a[:,:,0]
img=cv2.circle(b,(train_label[random][0],train_label[random][1]),2,(0,0,255),-1)
cv2.imwrite('test.png',b)




train=np.load(val_filename)
train_label=np.load(val_groundtruth)
random=100

a=np.around(train[random]*255.0+128.0)
b=np.empty_like(a)
b[:,:,0]=a[:,:,2]
b[:,:,1]=a[:,:,1]
b[:,:,2]=a[:,:,0]
img=cv2.circle(b,(train_label[random][0],train_label[random][1]),2,(0,0,255),-1)
cv2.imwrite('test2.png',b)