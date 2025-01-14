# -*- coding: UTF-8 -*- 
import math, json, os, sys
import glob
import random
import matplotlib.image as Img
import cv2
from scipy import misc
from keras.models import Sequential, model_from_json, Model, load_model
from keras.optimizers import SGD
from keras.layers import Input, Reshape, Concatenate, Activation, Flatten, merge
from keras.layers.normalization import BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D,AveragePooling2D, Dense, Dropout
from keras import initializers
from keras.callbacks import LearningRateScheduler,TensorBoard 
from keras import backend as K
from keras.utils import np_utils  
from keras.utils import plot_model
# from keras.utils.vis_utils import plot_model  



from keras.utils import np_utils
import numpy as np
import h5py

img_row = 1024
img_cols = 1024
batch_size = 4
momentum = 0.9
base_lr = 0.001
Lambda=0.5
nb_epoch = 100
epochs_drop = 20
gamma =  0.96

root = '/home/smiletranquilly/HeightEstimation/largeImage'
os.chdir(root)
dset = '/home/Dataset/Potsdam_1024'
Valdir = '/home/Dataset/Potsdam_1024_Val'

google_dir = './model/05-26'
log_path = './log/05-26/'

isExists=os.path.exists(google_dir)    
if not isExists:
    os.makedirs(google_dir) 
isExists=os.path.exists(log_path)
if not isExists:
    os.makedirs(log_path) 

def scale_invarient_error(y_true,y_pred):
    log_1=K.clip(y_pred,K.epsilon(),np.inf)+1. #K.log()
    log_2=K.clip(y_true,K.epsilon(),np.inf)+1. #K.log()
    return K.mean(K.square(log_1-log_2),axis=-1)-Lambda*K.square(K.mean(log_1-log_2,axis=-1))

def gen_batch(X1, X2, batch_size):

    while True:
        idx = np.random.choice(X1.shape[0], batch_size, replace=False) #random choice
        # X1.shape[0]为所有数据总量
        yield X1[idx], X2[idx]

def rescale(data):
    data=data.astype('float32')
    data /= 255.0   
    return data
        
def process_line(line):  
    tmp = Img.imread(line) 
    tmp = np.array(tmp)
    width = tmp.shape[1]
    x = tmp[:,:width//2,:]  #Data
    y = tmp[:,width//2:,:]  #label
    return x,y  
  
def generate_arrays_from_file(input_paths,batch_size):  
    while 1:  
        random.shuffle(input_paths)# every epoch shuffle  
        print("The length of inputs is %d"%len(input_paths))
        cnt = 0  
        X =[]  
        Y =[]  
        for line in input_paths:  
            # create Numpy arrays of input data  
            # and labels, from each line in the file  
            x, y = process_line(line)  
            X.append(x)  
            Y.append(y)  
            cnt += 1  
            if cnt==batch_size:  
                cnt = 0
                Y = np.array(Y) 
                Y = Y[:,:,:,0] # only take one channel!
                for i in range(2):
                    X = np.array([cv2.pyrDown(X[i]) for i in range(len(X))])
                    Y = np.array([cv2.pyrDown(Y[i]) for i in range(len(X))]) 
                X = rescale(X)
                Y = rescale(Y) 
                yield (X,Y)  
                X = []  
                Y = []  
     

def load_data(input_dir):
    if input_dir is None or not os.path.exists(input_dir):
        raise Exception("input_dir does not exist")
    # if DSM_dir is None or not os.path.exists(DSM_dir):
    #     raise Exception("DSM_dir does not exist")
    
    input_paths = glob.glob(os.path.join(input_dir, "*.jpg")) #返回所有匹配的文件路径列表
    # DSM_paths = glob.glob(os.path.join(DSM_dir, "*.jpg")) #返回所有匹配的文件路径列表
    if len(input_paths) == 0:
        input_paths = glob.glob(os.path.join(input_dir, "*.png")) # 没有jpg就看png
    # if len(DSM_paths) == 0:
    #     DSM_paths = glob.glob(os.path.join(DSM_dir, "*.png"))
    # input_paths = sorted(input_paths)
    # DSM_paths = sorted(DSM_paths)
    return input_paths,len(input_paths)


def google_net(model_name= 'modify_googlenet'):
    # channel last
    bias_ini = initializers.Constant(0.5)
    w_ini = initializers.glorot_uniform()
     
    inputs=Input(shape=(int(img_row/4),int(img_cols/4),3))
    #conv1/7x7_s2
    conv1=Convolution2D(64,(7,7),strides=(2,2),name='conv1/7x7_s2',padding='same',
                        init=w_ini,bias_initializer=bias_ini)(inputs)
    conv1=Activation('sigmoid')(conv1)
    conv1=MaxPooling2D(pool_size=(3, 3),strides=(2,2))(conv1)
    conv1=BatchNormalization(axis=-1)(conv1)

    #conv2/3x3_reduce
    conv2 = Convolution2D(64,(1,1),name='conv2/1x1',init=w_ini,bias_initializer=bias_ini)(conv1)
    conv2=Activation('sigmoid')(conv2)
    conv2 = Convolution2D(192,(3,3),name='conv2/3x3_reduce',padding='same',init=w_ini,bias_initializer=bias_ini)(conv2)
    conv2=Activation('sigmoid')(conv2)
    conv2=BatchNormalization(axis=-1)(conv2)
    conv2=MaxPooling2D(pool_size=(3, 3),strides=(2,2))(conv2)

    #inception_3a/1x1
    conv3_1 = Convolution2D(64,(1,1),name='inception_3a/1x1',init=w_ini,bias_initializer=bias_ini)(conv2)
    conv3_1 = Activation('sigmoid')(conv3_1)

    #inception_3a/3x3_reduce
    conv3_3 =  Convolution2D(96,(1,1),name='inception_3a/3x3_reduce',init=w_ini,bias_initializer=bias_ini)(conv2)
    conv3_3 = Activation('sigmoid')(conv3_3)
    conv3_3 =  Convolution2D(128,(3,3),name='inception_3a/3x3',padding='same',init=w_ini,bias_initializer=bias_ini)(conv3_3)
    conv3_3 = Activation('sigmoid')(conv3_3)

    # inception_3a/5x5_reduce
    conv3_5 =  Convolution2D(16,(1,1),name='inception_3a/5x5_reduce',init=w_ini,bias_initializer=bias_ini)(conv2)
    conv3_5 = Activation('sigmoid')(conv3_5)
    conv3_5 =  Convolution2D(32,(5,5),name='inception_3a/5x5',padding='same',init=w_ini,bias_initializer=bias_ini)(conv3_5)
    conv3_5 = Activation('sigmoid')(conv3_5)

    #inception_3a/pool
    conv3_p= MaxPooling2D(pool_size=(3, 3),strides=(1,1),padding='same',name='inception_3a/pool')(conv2)
    conv3_p =  Convolution2D(32,(1,1),name='inception_3a/pool_proj',init=w_ini,bias_initializer=bias_ini)(conv3_p)
    conv3_p = Activation('sigmoid')(conv3_p)

    # inception_3a/output
    inception_3a_output = Concatenate(axis = -1)([conv3_1,conv3_3,conv3_5,conv3_p])

    # inception_3b/1x1
    conv3b_1 = Convolution2D(128,(1,1),name='inception_3b/1x1',init=w_ini,bias_initializer=bias_ini)(inception_3a_output)
    conv3b_1 = Activation('sigmoid')(conv3b_1)

    #inception_3b/3x3_reduce
    conv3b_3 =  Convolution2D(128,(1,1),name='inception_3b/3x3_reduce',init=w_ini,bias_initializer=bias_ini)(inception_3a_output)
    conv3b_3 = Activation('sigmoid')(conv3b_3)
    conv3b_3 =  Convolution2D(192,(3,3),name='inception_3b/3x3',padding='same',init=w_ini,bias_initializer=bias_ini)(conv3b_3)
    conv3b_3 = Activation('sigmoid')(conv3b_3)

    # inception_3b/5x5_reduce
    conv3b_5 =  Convolution2D(32,(1,1),name='inception_3b/5x5_reduce',init=w_ini,bias_initializer=bias_ini)(inception_3a_output)
    conv3b_5 = Activation('sigmoid')(conv3b_5)
    conv3b_5 =  Convolution2D(96,(5,5),name='inception_3b/5x5',padding='same',init=w_ini,bias_initializer=bias_ini)(conv3b_5)
    conv3b_5 = Activation('sigmoid')(conv3b_5)

    #inception_3b/pool
    conv3b_p = MaxPooling2D(pool_size=(3, 3),strides=(1,1),padding='same',name='inception_3b/pool')(inception_3a_output)
    conv3b_p = Convolution2D(64,(1,1),name='inception_3b/pool_proj',init=w_ini,bias_initializer=bias_ini)(conv3b_p)
    conv3b_p = Activation('sigmoid')(conv3b_p)

    # inception_3b/output
    inception_3b_output = Concatenate(axis = -1)([conv3b_1,conv3b_3,conv3b_5,conv3b_p])

    #pool3/3x3_s2
    pool3_3 = MaxPooling2D(pool_size=(3, 3),strides=(2,2),padding='valid',name='pool3/3x3_s2')(inception_3b_output)

    # inception_4a/1x1
    conv4a_1 = Convolution2D(192,(1,1),name='inception_4a/1x1',init=w_ini,bias_initializer=bias_ini)(pool3_3)
    conv4a_1 = Activation('sigmoid')(conv4a_1)

    #inception_4a/3x3_reduce
    conv4a_3 =  Convolution2D(96,(1,1),name='inception_4a/3x3_reduce',init=w_ini,bias_initializer=bias_ini)(pool3_3)
    conv4a_3 = Activation('sigmoid')(conv4a_3)
    conv4a_3 =  Convolution2D(208,(3,3),name='inception_4a/3x3',padding='same',
                                init=w_ini,bias_initializer=bias_ini)(conv4a_3)
    conv4a_3 = Activation('sigmoid')(conv4a_3)

    # inception_4a/5x5_reduce
    conv4a_5 =  Convolution2D(16,(1,1),name='inception_4a/5x5_reduce',init=w_ini,bias_initializer=bias_ini)(pool3_3)
    conv4a_5 = Activation('sigmoid')(conv4a_5)
    conv4a_5 =  Convolution2D(48,(5,5),name='inception_4a/5x5',padding='same',init=w_ini,bias_initializer=bias_ini)(conv4a_5)
    conv4a_5 = Activation('sigmoid')(conv4a_5)

    #inception_4a/pool
    conv4a_p = MaxPooling2D(pool_size=(3, 3),strides=(1,1),padding='same',name='inception_4a/pool')(pool3_3)
    conv4a_p = Convolution2D(64,(1,1),name='inception_4a/pool_proj',init=w_ini,bias_initializer=bias_ini)(conv4a_p)
    conv4a_p = Activation('sigmoid')(conv4a_p)

    # inception_4a/output
    inception_4a_output = Concatenate(axis = -1)([conv4a_1,conv4a_3,conv4a_5,conv4a_p])

    #loss1/ave_pool
    loss1_ap = AveragePooling2D(pool_size=(5, 5),strides=(3,3),padding='valid',name='loss1/ave_pool')(inception_4a_output)

    #loss1/conv
    convloss1 = Convolution2D(128,(1,1),name='loss1/conv',init=w_ini,bias_initializer=bias_ini)(loss1_ap)
    convloss1 = Activation('sigmoid')(convloss1)

    #loss1/fc
    loss1_fc=Flatten(name='loss1/fl')(convloss1)
    loss1_fc=Dense((img_row/4)*(img_cols/4),use_bias=True,init=w_ini,bias_initializer=bias_ini)(loss1_fc) #256*256
    loss1_fc=Activation('sigmoid')(loss1_fc) 
    # loss1_fc=Activation('softmax')(loss1_fc)
    dsm_out = Reshape((int(img_row/4),int(img_cols/4)))(loss1_fc)

    google_net = Model(inputs=inputs,outputs=dsm_out,name = model_name)

    return google_net

def step_decay(epoch):
    return base_lr * math.pow (gamma ,math.floor(epoch / epochs_drop))

def train():
    # Create optimizers
    opt_dcgan = SGD(base_lr,momentum)
    inputs_path,train_num = load_data(dset)
    val_path,val_num = load_data(dset)
    batches = generate_arrays_from_file(inputs_path,batch_size=batch_size)
    val_batches = generate_arrays_from_file(inputs_path,batch_size=batch_size)
    
    google_model = google_net()
    tensorboard = TensorBoard(log_dir=log_path)
    lrate = LearningRateScheduler(step_decay)  
    google_model.compile(loss=scale_invarient_error,optimizer=opt_dcgan,metrics=['accuracy'])
    plot_model(google_model, to_file="./figures/Googlenet.png" , show_shapes=True, show_layer_names=True)
    print("Start training")
    #print net info
    google_model.summary()
    google_model.fit_generator(batches,samples_per_epoch=math.ceil(train_num/batch_size) ,nb_epoch=nb_epoch,
    callbacks=[tensorboard,lrate],validation_data=val_batches,validation_steps=math.ceil(val_num/batch_size))
    google_model.save(google_dir+'googlenet.h5')
    return

train()