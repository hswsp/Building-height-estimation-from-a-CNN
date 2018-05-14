# -*- coding: UTF-8 -*-   

import numpy as np
import pandas as pd
import cv2
import os
import time
import matplotlib.pyplot as plt
import hdf5storage
%matplotlib inline

from keras.models import Sequential, model_from_json, Model, load_model
from keras.optimizers import SGD
from keras.layers import Input, Reshape, concatenate, Activation, Flatten, merge,normalization
from keras.layers import Convolution2D, MaxPooling2D, Dense, Dropout
from keras.initializers import Constant
from keras.callbacks import LearningRateScheduler
from keras import backend as K
from keras.engine.topology import Merge

batch_size=100
epochs=500
# test_iter = 8
# test_interval = 1000

img_row=256
img_cols=256
learning_rate=0.1
momentum=0.9
Lambda=0.5
base_lr = 0.01
gamma =  0.96
stepsize = 100

dset = '/home/download/'
TrainData1='Potsdam.mat'
TrainData2='Potsdam1.mat'
ValData = 'Vaihingen.mat
google_dir = './largeImage/model'


def scale_invarient_error(y_true,y_pred):
    # log_1=K.log(K.clip(y_pred,K.epsilon(),np.inf)+1.)
    # log_2=K.log(K.clip(y_true,K.epsilon(),np.inf)+1.)
    # return K.mean(K.square(log_1-log_2),axis=-1)-Lambda*K.square(K.mean(log_1-log_2,axis=-1))
    dist = numpy.sqrt(numpy.sum(numpy.square(y_true - y_pred)))  
    return dist/(2.0*len(y_pred))


def normalization(X):
    #[0,255]=>[0,1]
    return X / 255.0

def normalization_float(X,maxX):
    return X / maxX


def load_largeData(X_depths,img_type):
    dsm_num = len(X_depths)
    print dsm_num
    if img_type == 'depths':
        X_depths1 = np.array(X_depths[:dsm_num/4]).astype(np.float16).reshape(-1,img_row*img_cols)# must match output of net
        X_depths2 = np.array(X_depths[dsm_num / 4:dsm_num / 2]).astype(np.float16).reshape(-1,img_row*img_cols)
        X1 = np.concatenate((X_depths1,X_depths2), axis = 0)
        del X_depths1,X_depths2
        X_depths3 = np.array(X_depths[dsm_num / 2:3 * dsm_num / 4]).astype(np.float16).reshape(-1,img_row*img_cols)
        X_depths4 = np.array(X_depths[3 * dsm_num / 4:]).astype(np.float16).reshape(-1,img_row*img_cols)
        X2 = np.concatenate((X_depths3, X_depths4), axis=0)
        del X_depths3, X_depths4
        X_depths_train = np.concatenate((X1, X2), axis=0)
    elif img_type == 'images':
        X_depths1 = np.array(X_depths[:dsm_num / 4])
        X_depths2 = np.array(X_depths[dsm_num / 4:dsm_num / 2])
        X1 = normalization(np.concatenate((X_depths1.astype(np.float16), X_depths2.astype(np.float16)), axis=0))
        del X_depths1, X_depths2
        X_depths3 = np.array(X_depths[dsm_num / 2:3 * dsm_num / 4])
        X_depths4 = np.array(X_depths[3 * dsm_num / 4:])
        X2 = normalization(np.concatenate((X_depths3.astype(np.float16), X_depths4.astype(np.float16)), axis=0))
        del X_depths3, X_depths4
        X_depths_train = np.concatenate((X1, X2), axis=0)

    return X_depths_train

def loadData():
    #channels_last
     with h5py.File(dset+TrainData1, "r") as hf: 
        X_depths_train = hf["depths"] # 关键：这里的h5f与dataset并不包含真正的数据，只是包含了数据的相关信息，不会占据内存空间
        X_depths_train = load_largeData(X_depths_train,'depths')

        X_images_train = hf["images"]
        X_images_train = load_largeData(X_images_train,'images')
        hf.close()

    with h5py.File(dset+TrainData2, "r") as hm:
        X_depths_train1 = hm["depths"]
        X_depths_train1 = load_largeData(X_depths_train1,'depths')
        X_images_train1 = hm["images"]
        X_images_train1 = load_largeData(X_images_train1,'images')
        hm.close()

    X_depths_train = np.concatenate((X_depths_train, X_depths_train1),axis = 0)
    X_depths_train = normalization_float(X_depths_train,np.max(X_depths_train))

    X_images_train = np.concatenate((X_images_train, X_images_train1),axis = 0)

    
    X_images_train = X_images_train.transpose(0, 2, 3, 1) # matlab->python= num*c*H*W


    with h5py.File(dset+ValData, "r") as hv:
            X_depths_val = hv["depths"]
            dsm_num = len(X_depths_val)
            X_depths_val = np.array(X_depths_val[:dsm_num / 8]).astype(np.float16).reshape(-1,img_row*img_cols)
            X_depths_val = normalization_float(depths_val,np.max(depths_val))

            X_images_val =hv["images"]
            img_num = len(X_images_val)
            X_images_val = np.array(X_images_val[:img_num / 8]).astype(np.uint8)
            
            X_images_val = X_images_val.transpose(0, 2, 3, 1)
            X_images_val = normalization(X_images_val.astype(np.float16))
    return X_depths_train, X_images_train, X_depths_val, X_images_val

def gen_batch(X1, X2, batch_size):

    while True:
        idx = np.random.choice(X1.shape[0], batch_size, replace=False) #random choice
        # X1.shape[0]为所有数据总量
        yield X1[idx], X2[idx]

def step_decay(epoch):
    return base_lr * math.pow (gamma ,math.floor(epoch / stepsize))

def google_net(model_name= 'modify_googlenet'):
    # channel last
    bias_ini = initializers.Constant(0.5)
     
    inputs=Input(shape=(int(img_row/2),int(img_cols/2),3))
    #conv1/7x7_s2
    conv1=Convolution2D(64,(7,7),strides=(2,2),name='conv1/7x7_s2',padding='same',bias_initializer=bias_ini)(inputs)
    conv1=Activation('relu')(conv1)
    conv1=MaxPooling2D(pool_size=(3, 3),strides=(2,2))(conv1)
    conv1=BatchNormalization(axis=-1)(conv1)

    #conv2/3x3_reduce
    conv2 = Convolution2D(64,(1,1),name='conv2/1x1',bias_initializer=bias_ini)(conv1)
    conv2=Activation('relu')(conv2)
    conv2 = Convolution2D(192,(3,3),name='conv2/3x3_reduce',padding='same',bias_initializer=bias_ini)(conv2)
    conv2=Activation('relu')(conv2)
    conv2=BatchNormalization(axis=-1)(conv2)
    conv2=MaxPooling2D(pool_size=(3, 3),strides=(2,2))(conv2)

    #inception_3a/1x1
    conv3_1 = Convolution2D(64,(1,1),name='inception_3a/1x1',bias_initializer=bias_ini)(conv2)
    conv3_1 = Activation('relu')(conv3_1)

    #inception_3a/3x3_reduce
    conv3_3 =  Convolution2D(96,(1,1),name='inception_3a/3x3_reduce',bias_initializer=bias_ini)(conv2)
    conv3_3 = Activation('relu')(conv3_3)
    conv3_3 =  Convolution2D(128,(3,3),name='inception_3a/3x3',padding='same',bias_initializer=bias_ini)(conv3_3)
    conv3_3 = Activation('relu')(conv3_3)

    # inception_3a/5x5_reduce
    conv3_5 =  Convolution2D(16,(1,1),name='inception_3a/5x5_reduce',bias_initializer=bias_ini)(conv2)
    conv3_5 = Activation('relu')(conv3_5)
    conv3_5 =  Convolution2D(32,(5,5),name='inception_3a/5x5',padding='same',bias_initializer=bias_ini)(conv3_5)
    conv3_5 = Activation('relu')(conv3_5)

    #inception_3a/pool
    conv3_p=MaxPooling2D(pool_size=(3, 3),strides=(1,1),padding='same',name='inception_3a/pool')(conv2)
    conv3_p =  Convolution2D(32,(1,1),name='inception_3a/pool_proj',bias_initializer=bias_ini)(conv3_p)
    conv3_p = Activation('relu')(conv3_p)

    # inception_3a/output
    inception_3a_output = Concatenate(axis = -1)([conv3_1,conv3_3,conv3_5,conv3_p])

    # inception_3b/1x1
    conv3b_1 = Convolution2D(128,(1,1),name='inception_3b/1x1',bias_initializer=bias_ini)(inception_3a_output)
    conv3b_1 = Activation('relu')(conv3b_1)

    #inception_3b/3x3_reduce
    conv3b_3 =  Convolution2D(128,(1,1),name='inception_3b/3x3_reduce',bias_initializer=bias_ini)(inception_3a_output)
    conv3b_3 = Activation('relu')(conv3b_3)
    conv3b_3 =  Convolution2D(192,(3,3),name='inception_3b/3x3',padding='same',bias_initializer=bias_ini)(conv3b_3)
    conv3b_3 = Activation('relu')(conv3b_3)

    # inception_3b/5x5_reduce
    conv3b_5 =  Convolution2D(32,(1,1),name='inception_3b/5x5_reduce',bias_initializer=bias_ini)(inception_3a_output)
    conv3b_5 = Activation('relu')(conv3b_5)
    conv3b_5 =  Convolution2D(96,(5,5),name='inception_3b/5x5',padding='same',bias_initializer=bias_ini)(conv3b_5)
    conv3b_5 = Activation('relu')(conv3b_5)

    #inception_3b/pool
    conv3b_p = MaxPooling2D(pool_size=(3, 3),strides=(1,1),padding='same',name='inception_3b/pool')(inception_3a_output)
    conv3b_p = Convolution2D(64,(1,1),name='inception_3b/pool_proj',bias_initializer=bias_ini)(conv3b_p)
    conv3b_p = Activation('relu')(conv3b_p)

    # inception_3b/output
    inception_3b_output = Concatenate(axis = -1)(conv3b_1,conv3b_3,conv3b_5,conv3b_p)

    #pool3/3x3_s2
    pool3_3 = MaxPooling2D(pool_size=(3, 3),strides=(2,2),padding='valid',name='pool3/3x3_s2')(inception_3b_output)

    # inception_4a/1x1
    conv4a_1 = Convolution2D(192,(1,1),name='inception_4a/1x1',bias_initializer=bias_ini)(pool3_3)
    conv4a_1 = Activation('relu')(conv4a_1)

    #inception_4a/3x3_reduce
    conv4a_3 =  Convolution2D(96,(1,1),name='inception_4a/3x3_reduce',bias_initializer=bias_ini)(pool3_3)
    conv4a_3 = Activation('relu')(conv4a_3)
    conv4a_3 =  Convolution2D(208,(3,3),name='inception_4a/3x3',padding='same',bias_initializer=bias_ini)(conv4a_3)
    conv4a_3 = Activation('relu')(conv4a_3)

    # inception_4a/5x5_reduce
    conv4a_5 =  Convolution2D(16,(1,1),name='inception_4a/5x5_reduce',bias_initializer=bias_ini)(pool3_3)
    conv4a_5 = Activation('relu')(conv4a_5)
    conv4a_5 =  Convolution2D(48,(5,5),name='inception_4a/5x5',padding='same',bias_initializer=bias_ini)(conv4a_5)
    conv4a_5 = Activation('relu')(conv4a_5)

    #inception_4a/pool
    conv4a_p = MaxPooling2D(pool_size=(3, 3),strides=(1,1),padding='same',name='inception_4a/pool')(pool3_3)
    conv4a_p = Convolution2D(64,(1,1),name='inception_4a/pool_proj',bias_initializer=bias_ini)(conv4a_p)
    conv4a_p = Activation('relu')(conv4a_p)

    # inception_4a/output
    inception_4a_output = Concatenate(axis = -1)([conv4a_1,conv4a_3,conv4a_5,conv4a_p])

    #loss1/ave_pool
    loss1_ap = AveragePooling2D(pool_size=(5, 5),strides=(3,3),padding='valid',name='loss1/ave_pool')(inception_4a_output)

    #loss1/conv
    convloss1 = Convolution2D(128,(1,1),name='loss1/conv',bias_initializer=bias_ini)(loss1_ap)
    convloss1 = Activation('relu')(convloss1)

    #loss1/fc
    loss1_fc=Flatten(name='loss1/fl')(convloss1)
    loss1_fc=Dense(16384,use_bias=True,bias_initializer=bias_ini)(loss1_fc) #128*128
    loss1_fc=Activation('relu')(loss1_fc) 

    google_net = Model(input=inputs,output=loss1_fc,name = model_name)

    return google_net

    

def train():

    google_model = google_net()
    google_model.compile(loss=scale_invarient_error,optimizer=SGD(learning_rate,momentum,decay=0.0),metrics=['accuracy'])
    google_model.summary()

    # Load and rescale data
    X_full_train, X_sketch_train, X_full_val, X_sketch_val = data_utils.load_data(dset, image_data_format)

    # for e in range(nb_epoch):
    # stop at epochs
    lrate = LearningRateScheduler(step_decay)
    google_model.fit(X_train,y_train,epochs=epochs,callbacks=[lrate],batch_size=batch_size,shuffle=True,validation_data=(X_images_val,X_depths_val) ) 
    # steps_per_epoch =,validation_steps = test_iter                       
    #save_model
    model.save(google_dir)
















