# -*- coding: UTF-8 -*-   
import numpy as np
import pandas as pd
import cv2
import os
import time
import matplotlib.pyplot as plt
import hdf5storage
import h5py
import math
#%matplotlib inline
import time
from keras.models import Sequential, model_from_json, Model, load_model
from keras.optimizers import SGD
from keras.layers import Input, Reshape, Concatenate, Activation, Flatten, merge
from keras.layers.normalization import BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D,AveragePooling2D, Dense, Dropout
from keras import initializers
from keras.callbacks import LearningRateScheduler,TensorBoard 
from keras import backend as K


batch_size=100
epochs=1000
img_row=1024
img_cols=1024
momentum=0.9
Lambda=0.5
base_lr = 0.01
gamma =  0.96
stepsize = 100

root = '/home/smiletranquilly/HeightEstimation/'
dset = '/home/Dataset/'
TrainData1='Potsdam_1024.mat'
# TrainData2='Postdam1.mat'
ValData = 'Vaihingen_1024.mat'
os.chdir(root)

google_dir = './largeImage/model-5-18/'
log_filepath = './largeImage/log-5-18/'

isExists=os.path.exists(google_dir)    
if not isExists:
    os.makedirs(google_dir) 
isExists=os.path.exists(log_filepath)
if not isExists:
    os.makedirs(log_filepath) 


def step_decay(epoch):
    return base_lr * math.pow (gamma ,math.floor(epoch / stepsize))

# loss function
def scale_invarient_error(y_true,y_pred):
    # log_1=K.log(K.clip(y_pred,K.epsilon(),np.inf)+1.)
    # log_2=K.log(K.clip(y_true,K.epsilon(),np.inf)+1.)
    # return K.mean(K.square(log_1-log_2),axis=-1)-Lambda*K.square(K.mean(log_1-log_2,axis=-1))
    dist = K.sqrt(K.sum(K.square(y_true - y_pred)))  
    return dist/(2.0*(img_row*img_cols)/4)



def normalization(X):
    #[0,255]=>[0,1]
    X = X.data.astype('float32')
    return X / 255.0

def normalization_float(X,maxX):
    return X / maxX


def load_largeData(X_depths,img_type):
    dsm_num = len(X_depths)
    print dsm_num
    if img_type == 'depths':
        # not norm
        X_depths1 = np.array(X_depths[:dsm_num/4]).astype(np.float32)
        X_depths2 = np.array(X_depths[dsm_num / 4:dsm_num / 2]).astype(np.float32)
        X1 = np.concatenate((X_depths1,X_depths2), axis = 0)
        del X_depths1,X_depths2
        X_depths3 = np.array(X_depths[dsm_num / 2:3 * dsm_num / 4]).astype(np.float32)
        X_depths4 = np.array(X_depths[3 * dsm_num / 4:]).astype(np.float32)
        X2 = np.concatenate((X_depths3, X_depths4), axis=0)
        del X_depths3, X_depths4
        X_depths_train = np.concatenate((X1, X2), axis=0)
        X_depths_train = np.array(X_depths_train)
    elif img_type == 'images':
        # have been norm
        X_depths1 = np.array(X_depths[:dsm_num / 4]).astype(np.float32)
        X_depths2 = np.array(X_depths[dsm_num / 4:dsm_num / 2]).astype(np.float32) # must be float32 in order to divide
        X1 = normalization(np.concatenate((X_depths1, X_depths2), axis=0))
        del X_depths1, X_depths2
        X_depths3 = np.array(X_depths[dsm_num / 2:3 * dsm_num / 4]).astype(np.float32)
        X_depths4 = np.array(X_depths[3 * dsm_num / 4:]).astype(np.float32)
        X2 = normalization(np.concatenate((X_depths3, X_depths4), axis=0))
        del X_depths3, X_depths4
        X_depths_train = np.concatenate((X1, X2), axis=0)
        X_depths_train = np.array(X_depths_train)

    return X_depths_train

def loadData(dset):
    #channels_last
    with h5py.File(dset+TrainData1, "r") as hf: 
        X_depths_train = hf["depths"] # 关键：这里的h5f与dataset并不包含真正的数据，只是包含了数据的相关信息，不会占据内存空间         
        X_depths_train = load_largeData(X_depths_train,'depths')
        # X_depths_train1 = np.array(X_depths_train1[:len(X_depths_train1)]).astype(np.float32)
        print X_depths_train.shape
        X_images_train = hf["images"]
        X_images_train = load_largeData(X_images_train,'images')
        # X_images_train1 = np.array(X_images_train1[:len(X_images_train1)]).astype(np.float32)
        X_depths_train = normalization(X_depths_train)
        hf.close()

    # with h5py.File(dset+TrainData2, "r") as hm:#'test.mat'
    #     X_depths_train = hm["depths"]
    #     X_depths_train = load_largeData(X_depths_train,'depths')
    #     print X_depths_train.shape
    #     X_images_train = hm["images"]
    #     X_images_train = load_largeData(X_images_train,'images')
    #     hm.close()
    
    # X_depths_train = np.concatenate((X_depths_train, X_depths_train1),axis = 0)
    # X_depths_train = normalization(X_depths_train)

    # X_images_train = np.concatenate((X_images_train, X_images_train1),axis = 0) 
    # X_images_train = X_images_train.transpose(0, 2, 3, 1) # matlab->python= num*c*H*W
    
    with h5py.File(dset+ValData, "r") as hv:#'test_val.mat'
            X_depths_val = hv["depths"]
            dsm_num_val = len(X_depths_val)
            print dsm_num_val
            X_depths_val = np.array(X_depths_val[:dsm_num_val / 2]).astype(np.float32)
            X_depths_val =  normalization(X_depths_val)# normalization_float(X_depths_val,np.max(X_depths_val)) 

            X_images_val =hv["images"]
            img_num = len(X_images_val)
            X_images_val = np.array(X_images_val[:img_num / 2])
            
            X_images_val = X_images_val.transpose(0, 2, 3, 1)
            X_images_val = normalization(X_images_val.astype(np.float32))
    return X_depths_train, X_images_train, X_depths_val, X_images_val

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
    conv3_p= MaxPooling2D(pool_size=(3, 3),strides=(1,1),padding='same',name='inception_3a/pool')(conv2)
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
    inception_3b_output = Concatenate(axis = -1)([conv3b_1,conv3b_3,conv3b_5,conv3b_p])

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
    loss1_fc=Dense(img_row*img_cols/4,use_bias=True,bias_initializer=bias_ini)(loss1_fc) #128*128
    loss1_fc=Activation('relu')(loss1_fc) 
    dsm_out = Reshape((int(img_row/2),int(img_cols/2)))(loss1_fc)

    google_net = Model(input=inputs,output=loss1_fc,name = model_name)

    return google_net

def rescale(data):
    data=data.astype('float32')
    data /= 255.0    
    return data

def pred_single_image_depth_using_CNN(path):
    model=load_model(google_dir+'googlenet_weights.h5',custom_objects={'scale_invarient_error':scale_invarient_error})
    img_array=cv2.imread(path)
    
    img_array=np.expand_dims(img_array,axis=0)
    img_array=np.array([cv2.resize(img_array[i],(img_row/2,img_cols/2)) for i in range(1)])
    img_array=np.array([cv2.pyrDown(img_array[i]) for i in range(1)])
    img_array=rescale(img_array)
    out = model.predict(img_array)
    return out

def eval(eval_dir):
    #load_model
    model=load_model(eval_dir,custom_objects={'scale_invarient_error':scale_invarient_error})
    print(model.evaluate(X_test[:100],y_test[:100]))  
    

def train():

    google_model = google_net()
    google_model.compile(loss=scale_invarient_error,optimizer=SGD(base_lr,momentum,decay=0.0),metrics=['accuracy'])
    google_model.summary()

    # Load and rescale data
    y_data, X_data, X_depths_val, X_images_val = loadData(dset)
    # y_data = normalization_float(y_data,np.max(y_data))
    
    
    img_num = len(X_data)
    train_end=int(0.9*img_num)
    test_num= img_num - train_end
    X_train=X_data[:train_end]
    y_train=y_data[:train_end]

    X_test=X_data[train_end:img_num]
    y_test=y_data[train_end:img_num]
    
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    print(X_images_val.shape)
    print(X_depths_val.shape)
    
    X_train=np.array([cv2.pyrDown(X_train[i]) for i in range(train_end)]) # input must in [0,1]!and type=float32
    y_train=np.array([cv2.pyrDown(y_train[i]) for i in range(train_end)])
    
    X_test=np.array([cv2.pyrDown(X_test[i]) for i in range(test_num)])
    y_test=np.array([cv2.pyrDown(y_test[i]) for i in range(test_num)])
    
    X_images_val=np.array([cv2.pyrDown(X_images_val[i]) for i in range(len(X_images_val))])
    X_depths_val=np.array([cv2.pyrDown(X_depths_val[i]) for i in range(len(X_depths_val))])
    
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    print(X_images_val.shape)
    print(X_depths_val.shape)
    
    # dim = img_row*img_cols/4
    # y_train = y_train.reshape(-1,dim)    # must match output of net                 
    # y_test = y_test.reshape(-1,dim)      
    # X_depths_val = X_depths_val.reshape(-1,dim)
                         

    # for e in range(nb_epoch):
    # stop at epochs
    lrate = LearningRateScheduler(step_decay)
    start = time.time()
    google_model.fit(X_train,y_train,epochs=epochs,callbacks=[lrate,TensorBoard(log_dir=log_filepath)],batch_size=batch_size,shuffle=True,validation_data=(X_test,y_test) ) 
    # steps_per_epoch =,validation_steps = test_iter                       
    #save_model
    google_model.save(google_dir+'googlenet_weights.h5')

train()













