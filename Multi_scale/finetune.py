# -*- coding: UTF-8 -*-
import numpy as np
import math, json, os, sys
import cv2
import time
#import matplotlib.pyplot as plt
import hdf5storage
import scipy.io as scio
import h5py
# %matplotlib inline
from keras.models import Sequential, model_from_json, Model, load_model
from keras.optimizers import SGD
from keras.layers import Input, Reshape, concatenate, Activation, Flatten, merge
from keras.layers import Convolution2D, MaxPooling2D, Dense, Dropout
from keras import backend as K
from keras.callbacks import TensorBoard,ReduceLROnPlateau,LearningRateScheduler

from keras.models import load_model

#设置当前目录
root = '/home/smiletranquilly/Multi-Scale_Deep_Network' 
os.chdir(root)

coarse_dir='./coarse_data/building_coarse_model_523.h5'
fine_dir='./fine_data/building_fine_model_523.h5'

log_corsepath = './log/building_corse_log_523'
log_finepath = './log/building_fine_log_523'

dataFile='/home/Dataset/Vaihingen_1024.mat' #Potsdam_1024.mat
dataFile2 = '/home/Dataset/Vaihingen_1024.mat'

base_model_corse = './coarse_data/building_coarse_model_522.h5'
base_model_fine = './fine_data/building_fine_model_522.h5'
# 新建文件夹
isExists=os.path.exists('coarse_data')    
if not isExists:
    # 如果不存在则创建目录
    os.makedirs('coarse_data')
    
isExists=os.path.exists('fine_data')
if not isExists:
    os.makedirs('fine_data')
    
isExists=os.path.exists('log')
if not isExists:
    os.makedirs('log')

batch_size=32
coarse_epochs = 1
fine_epoches = 1
img_row=1024
img_cols=1024
learning_rate=0.001
momentum=0.9
Lambda=0.5
stepsize = 100
base_lr = 0.001
gamma = 0.5

def step_decay(epoch):
    return base_lr * math.pow (gamma ,math.floor(epoch / stepsize))

def scale_invarient_error(y_true,y_pred):
    log_1=K.log(K.clip(y_pred,K.epsilon(),np.inf)+1.)
    log_2=K.log(K.clip(y_true,K.epsilon(),np.inf)+1.)
    return K.mean(K.square(log_1-log_2),axis=-1)-Lambda*K.square(K.mean(log_1-log_2,axis=-1))

def rescale(data):
    data=data.astype('float32')
    data /= 255.0   
    return data


def train_coarse():
    new_model = load_model(base_model_corse,custom_objects={'scale_invarient_error':scale_invarient_error})
    try:
        os.makedirs(log_corsepath)
    except:
        pass
    #将loss ，acc， val_loss ,val_acc记录tensorboard
    tensorboard = TensorBoard(log_dir=log_corsepath)#, histogram_freq=1,write_graph=True,write_images=1
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=20) 
    lrate = LearningRateScheduler(step_decay)                      
    new_model.fit(X_train,y_train,epochs=coarse_epochs,batch_size=batch_size,shuffle=True,validation_split=0.2,
    callbacks=[tensorboard,lrate])
                               
    #save_model
    new_model.save(coarse_dir)

def train_fine():
    #load_coarse_model:
    model=load_model(coarse_dir,custom_objects={'scale_invarient_error':scale_invarient_error})
    
    for layers in model.layers:
        layers.trainable=False
    
    #fine_model
    inputs=model.input
    
    #fine_1:
    fine_1=Convolution2D(63,(9,9),strides=(2,2),padding='same',name='fine_con2d1')(inputs)
    fine_1=Activation('relu',name='fine_relu1')(fine_1)
    fine_1=MaxPooling2D(pool_size=(2,2),name='fine_maxp1')(fine_1)
    
    #fine_2:
    coarse_output=model.output
    coarse_output=Reshape((int(img_row/8),int(img_cols/8),1))(coarse_output)
    fine_2=concatenate([fine_1,coarse_output],axis=3)
    
    #fine_3:
    fine_3=Convolution2D(64,(5,5),padding='same',name='fine_con2d2')(fine_2)
    fine_3=Activation('relu')(fine_3)
    
    #fine_4:
    fine_4=Convolution2D(1,(5,5),padding='same',name='fine_con2d3')(fine_3)
    fine_4=Activation('linear')(fine_4)
    fine_4=Reshape((int(img_row/8),int(img_cols/8)))(fine_4)
    
    
    model=Model(input=inputs,output=fine_4)
    model.compile(loss=scale_invarient_error,optimizer=SGD(learning_rate,momentum),metrics=['accuracy'])
    
    model.summary()
    
    try:
        os.makedirs(log_finepath)
    except:
        pass
    #将loss ，acc， val_loss ,val_acc记录tensorboard
    lrate = LearningRateScheduler(step_decay)
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=20)
    tensorboard = TensorBoard(log_dir=log_finepath) #, histogram_freq=1,write_graph=True,write_images=1
    history = model.fit(X_train,y_train,batch_size=batch_size,epochs=fine_epoches,shuffle=True,validation_split=0.2,
    callbacks=[tensorboard,lrate])
    
    #save model
    model.save(fine_dir)


with h5py.File(dataFile, "r") as mat:
    # number of the first dim
    X_data = mat['images'][:].transpose((0,2, 3, 1))
    print("length of X_data is %d" % len(X_data))
    y_data = mat['depths'][:]
    print("length of y_data is %d" % len(y_data))
with h5py.File(dataFile2, "r") as f:
    # number of the first dim
    X_data2 = f['images'][:].transpose((0,2, 3, 1))
    print("length of X_data is %d" % len(X_data2))
    y_data2 = f['depths'][:]
    print("length of y_data is %d" % len(y_data2))

X_data = np.concatenate((X_data,X_data2),axis=0)
y_data = np.concatenate((y_data,y_data2),axis=0)

X_data=rescale(X_data)
y_data=rescale(y_data)
# # Y轴镜像
# X_data = np.concatenate((X_data,X_data[:,::-1]),axis = 0)
# y_data = np.concatenate((y_data,y_data[:,::-1]),axis = 0)

image_num = len(X_data) 
depth_num = len(y_data)
try:
    image_num == depth_num
except IOError:
    print "number not match, input error!"
print image_num

train_end=int(0.8*image_num)
test_num= image_num - train_end
X_train=X_data[:train_end]
y_train=y_data[:train_end]
X_test=X_data[train_end:image_num]
y_test=y_data[train_end:image_num]
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
del X_data,y_data

# [1024,1024]->[512,512]

X_train=np.array([cv2.pyrDown(X_train[i]) for i in range(train_end)])
y_train=np.array([cv2.pyrDown(y_train[i]) for i in range(train_end)])
X_test=np.array([cv2.pyrDown(X_test[i]) for i in range(test_num)])
y_test=np.array([cv2.pyrDown(y_test[i]) for i in range(test_num)])

for j in range(2):
    # lable[128*128]   
    y_train=np.array([cv2.pyrDown(y_train[i]) for i in range(train_end)])
    y_test=np.array([cv2.pyrDown(y_test[i]) for i in range(test_num)])
    
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

print "start train coarse"
train_coarse()
print "start train fine"
train_fine()





