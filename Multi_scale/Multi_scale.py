# -*- coding: UTF-8 -*- 
import math, json, os, sys
import numpy as np
#import pandas as pd
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

#设置当前目录
root = '/home/smiletranquilly/Multi-Scale_Deep_Network' 
os.chdir(root)

coarse_dir='./coarse_data/coarse_model_523.h5'
coarse_weights='./coarse_data/coarse_weights_523.h5'
fine_dir='./fine_data/fine_model_523.h5'
fine_weights='./fine_data/fine_weights_523.h5'

log_corsepath = './log/building_corse_log_523'
log_finepath = './log/building_fine_log_523'

dataFile='/home/Dataset/Potsdam_1024.mat'
dataFile2 = '/home/Dataset/Vaihingen_1024.mat'

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

def scale_invarient_error(y_true,y_pred):
    log_1=K.log(K.clip(y_pred,K.epsilon(),np.inf)+1.)
    log_2=K.log(K.clip(y_true,K.epsilon(),np.inf)+1.)
    return K.mean(K.square(log_1-log_2),axis=-1)-Lambda*K.square(K.mean(log_1-log_2,axis=-1))

batch_size=32
coarse_epochs = 500
fine_epoches = 800
img_row=1024
img_cols=1024
# learning_rate=0.001
momentum=0.9
Lambda=0.5
stepsize = 100
base_lr = 0.001
gamma = 0.96

def step_decay(epoch):
    return base_lr * math.pow (gamma ,math.floor(epoch / stepsize))

def pred_single_image_depth_using_fine(path):
    model=load_model(fine_dir,custom_objects={'scale_invarient_error':scale_invarient_error})
    img_array=cv2.imread(path)
    img_array=np.expand_dims(img_array,axis=0)
    img_array=np.array([cv2.resize(img_array[i],(480,640)) for i in range(1)])
    img_array=np.array([cv2.pyrDown(img_array[i]) for i in range(1)])
    img_array=rescale(img_array)
    out=model.predict(img_array)
    return out

def pred_single_image_depth_using_coarse(path):
    model=load_model(coarse_dir,custom_objects={'scale_invarient_error':scale_invarient_error})
    img_array=cv2.imread(path)
    img_array=np.expand_dims(img_array,axis=0)
    img_array=np.array([cv2.resize(img_array[i],(480,640)) for i in range(1)])
    img_array=np.array([cv2.pyrDown(img_array[i]) for i in range(1)])
    img_array=rescale(img_array)
    out=model.predict(img_array)
    return out

def pred_single_image_depth_using_coarse_array(image_array):
    model=load_model(coarse_dir,custom_objects={'scale_invarient_error':scale_invarient_error})
    image_array=np.expand_dims(image_array,axis=0)
    image_array=np.array([cv2.resize(image_array[i],(480,640)) for i in range(1)])
    image_array=np.array([cv2.pyrDown(image_array[i]) for i in range(1)])
    image_array=rescale(image_array)
    out=model.predict(image_array)
    return out

def pred_single_image_depth_using_fine_array(image_array):
    model=load_model(fine_dir,custom_objects={'scale_invarient_error':scale_invarient_error})
    image_array=np.expand_dims(image_array,axis=0)
    image_array=np.array([cv2.resize(image_array[i],(480,640)) for i in range(1)])
    image_array=np.array([cv2.pyrDown(image_array[i]) for i in range(1)])
    image_array=rescale(image_array)
    out=model.predict(image_array)
    return out
def display_image(path):
    img_array=plt.imread(path)
    img_array=np.expand_dims(img_array,axis=0) 
    img_array=np.array([cv2.resize(img_array[i],(480,640)) for i in range(1)])
    img_array=rescale(img_array)
    plt.imshow(img_array[0])
    
def eval(eval_dir):
    #load_model
    model=load_model(eval_dir,custom_objects={'scale_invarient_error':scale_invarient_error})
    print(model.evaluate(X_test,y_test))  
    
def rescale(data):
    data=data.astype('float32')
    data /= 255.0   
    return data

def rescale_float(label):
    maxnum = np.max(label)
    label=label.astype('float32')
    label = label /255.0
    return label


def convert(mat,start,end):
# input must be matlab mat!
    X=[]
    y=[]
    img = mat['images'] #image
    depths = mat['depths'] # raw depths
    for i in range(start, end): # include left not right!
                img1 = img[i,...].transpose((1, 2, 0))
                img2 = depths[i]     #np.transpose() 
                X.append(img1)
                y.append(img2)
    return np.array(X),np.array(y)

def train_coarse():
    inputs=Input(shape=(int(img_row/2),int(img_cols/2),3))

    #coarse_1
    coarse_1=Convolution2D(96,(11,11),strides=(4,4),padding='same')(inputs)
    coarse_1=Activation('relu')(coarse_1)
    coarse_1=MaxPooling2D(pool_size=(2, 2))(coarse_1)

    #coarse_2
    coarse_2=Convolution2D(256,(5,5),padding='same')(coarse_1)
    coarse_2=Activation('relu')(coarse_2)
    coarse_2=MaxPooling2D(pool_size=(2,2))(coarse_2)

    #coarse_3
    coarse_3=Convolution2D(384,(3,3),padding='same')(coarse_2)
    coarse_3=Activation('relu')(coarse_3)

    #coarse_4
    coarse_4=Convolution2D(384,(3,3),padding='same')(coarse_3)
    coarse_4=Activation('relu')(coarse_4)

    #coarse_5
    coarse_5=Convolution2D(256,(3,3),padding='same',)(coarse_4)
    coarse_5=Activation('relu')(coarse_5)
    coarse_5=MaxPooling2D(pool_size=(2,2))(coarse_5)

    #coarse_6
    coarse_6=Flatten(name='coarse_6')(coarse_5)
    coarse_6=Dense(4096)(coarse_6)
    coarse_6=Activation('relu')(coarse_6)
    coarse_6=Dropout(0.5)(coarse_6)

    # Coarse 7
    coarse_7=Dense((int(img_row/8))*(int(img_cols/8)))(coarse_6)
    coarse_7=Activation('linear')(coarse_7)
    coarse_7=Reshape((int(img_row/8),int(img_cols/8)))(coarse_7)
        
    model=Model(input=inputs,output=coarse_7)
    model.compile(loss=scale_invarient_error,optimizer=SGD(base_lr,momentum),metrics=['accuracy'])
    
    #print model
    model.summary() 
    
    try:
        os.makedirs(log_corsepath)
    except:
        pass
    #将loss ，acc， val_loss ,val_acc记录tensorboard
    tensorboard = TensorBoard(log_dir=log_corsepath)#, histogram_freq=1,write_graph=True,write_images=1
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=20)
    lrate = LearningRateScheduler(step_decay)                       
    model.fit(X_train,y_train,epochs=coarse_epochs,batch_size=batch_size,shuffle=True,validation_split=0.2,
    callbacks=[tensorboard,lrate])
                               
    #save_model
    model.save(coarse_dir)
 

def train_fine():
    #load_coarse_model:
    model=load_model(coarse_dir,custom_objects={'scale_invarient_error':scale_invarient_error})
    
    for layers in model.layers:
        layers.trainable=False
    
    #fine_model
    inputs=model.input
    
    #fine_1:
    fine_1=Convolution2D(63,(9,9),strides=(2,2),padding='same')(inputs)
    fine_1=Activation('relu')(fine_1)
    fine_1=MaxPooling2D(pool_size=(2,2))(fine_1)
    
    #fine_2:
    coarse_output=model.output
    coarse_output=Reshape((int(img_row/8),int(img_cols/8),1))(coarse_output)
    fine_2=concatenate([fine_1,coarse_output],axis=3)
    
    #fine_3:
    fine_3=Convolution2D(64,(5,5),padding='same')(fine_2)
    fine_3=Activation('relu')(fine_3)
    
    #fine_4:
    fine_4=Convolution2D(1,(5,5),padding='same')(fine_3)
    fine_4=Activation('linear')(fine_4)
    fine_4=Reshape((int(img_row/8),int(img_cols/8)))(fine_4)
    
    
    model=Model(input=inputs,output=fine_4)
    model.compile(loss=scale_invarient_error,optimizer=SGD(base_lr,momentum),metrics=['accuracy'])
    
    model.summary()
    
    try:
        os.makedirs(log_finepath)
    except:
        pass
    #将loss ，acc， val_loss ,val_acc记录tensorboard
    lrate = LearningRateScheduler(step_decay)
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=20)
    tensorboard = TensorBoard(log_dir=log_finepath) #, histogram_freq=1,write_graph=True,write_images=1
    history = model.fit(X_train,y_train,batch_size=batch_size,epochs=fine_epoches,shuffle=True,
    validation_split=0.2,callbacks=[tensorboard,lrate])
    
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

#     print(X_1.shape,y_1.shape)
#     print(X_2.shape,y_2.shape)
#     print(X_3.shape,y_3.shape)
#     print(X_4.shape,y_4.shape)
#     X_5=np.concatenate((X_1,X_2),axis=0)
#     # release memory
#     del X_1,X_2
#     y_5=np.concatenate((y_1,y_2),axis=0)
#     del y_1,y_2

#     X_6=np.concatenate((X_4,X_3),axis=0)
#     del X_4,X_3
#     y_6=np.concatenate((y_4,y_3),axis=0)    
#     del y_4,y_3

X_data = np.concatenate((X_data,X_data2),axis=0)
y_data = np.concatenate((y_data,y_data2),axis=0)
#     del y_5,y_6
#     print(X_data.shape,y_data.shape)
# 归一化
X_data=rescale(X_data)
y_data=rescale(y_data)#rescale_float
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
    # X_train=np.array([cv2.pyrDown(X_train[i]) for i in range(train_end)])
    # y_train=np.array([cv2.pyrDown(y_train[i]) for i in range(train_end)])
    # y_test=np.array([cv2.pyrDown(y_test[i]) for i in range(test_num)])   
    # y_test=np.array([cv2.pyrDown(y_test[i]) for i in range(test_num)])

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

train_coarse()
train_fine()

