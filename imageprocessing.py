#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 20:58:10 2020

@author: rkaratt
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import PIL
from PIL import Image
import os
import io
data=[]
labels=[]

height = 128
width = 128
channels = 3
classes = 43
n_inputs = height * width*channels

for i in range(classes) :
    path = "/home/rkaratt/TeamProject/Training_Images/GTSRB/Final_Training/Images/{:05d}".format(i)
    print(path)
    Class=os.listdir(path)
    csv_dir = "/home/rkaratt/TeamProject/Training_Images/GTSRB/Final_Training/Images_Notation/GT-{:05d}.csv".format(i)
    print(csv_dir)
    df = pd.read_csv(csv_dir , index_col=0 , sep=';')
    #for a in Class[:-1]:
    for a in Class :
        try:
            image = cv2.imread(path+'/'+a)
            #print(a)
           # plt.imshow(image)

            cropped_image = image[df.loc[a,'Roi.Y1']:df.loc[a,'Roi.Y2'], df.loc[a,'Roi.X1']:df.loc[a,'Roi.X2']]
            #plt.imshow(cropped_image)
            #roi = im[y1:y2, x1:x2]
            # (x1,y1) as the top-left vertex and (x2,y2) as the bottom-right vertex

            #image_from_array = Image.fromarray(cropped_image, 'RGB')
            img = cv2.cvtColor(cropped_image,cv2.COLOR_BGR2RGB)
            image_from_array = Image.fromarray(img )
            #image_from_array.size
            plt.imshow(image_from_array)


            size_image = image_from_array.resize((height, width))
            size_image.save('/home/rkaratt/TeamProject/Training_Images/GTSRB/Final_Training/New_Img/{:05d}.ppm'.format(i)+a)
            plt.imshow(size_image)
            
            data.append(np.array(size_image))
            labels.append(i)
        except AttributeError:
            print(" ")
      
            
Cells=np.array(data)
labels=np.array(labels)

#Randomize the order of the input images
s=np.arange(Cells.shape[0])
np.random.seed(43)
np.random.shuffle(s)
Cells=Cells[s]
labels=labels[s]
Cells.shape


#Spliting the images into train and validation sets
(X_train,X_val)=Cells[(int)(0.2*len(labels)):],Cells[:(int)(0.2*len(labels))]
X_train = X_train.astype('float32')/255 
X_val = X_val.astype('float32')/255
(y_train,y_val)=labels[(int)(0.2*len(labels)):],labels[:(int)(0.2*len(labels))]
X_train.shape
X_val.shape
y_train.shape
y_val.dtype


#Using one hote encoding for the train and validation labels
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train, 43)
y_val = to_categorical(y_val, 43)
y_train.dtype
