# -*- coding: UTF-8 -*- 
import math, json, os, sys
import glob
import random
import matplotlib.image as Img
import cv2
from scipy import misc

import keras
from keras.utils import generic_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint,TensorBoard,LearningRateScheduler
from keras.models import Model
from keras.optimizers import Adam,SGD
from keras.layers.core import Flatten, Dense, Dropout, Activation, Lambda, Reshape
from keras.layers.convolutional import Conv2D, Deconv2D, ZeroPadding2D, UpSampling2D
from keras.layers import Input, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras import backend as K


from keras.utils import np_utils
import numpy as np
import h5py

img_row = 1024
img_cols = 1024
batch_size = 4
Val_batch_size = 16
momentum = 0.9  
base_lr = 0.01
Lambda=100
nb_epoch = 60
epochs_drop = 16
gamma =  0.5

root = '/home/smiletranquilly/HeightEstimation/FCRN'
os.chdir(root)
dset = '/home/Dataset/Vaihingen_1024_merge'
Valdir = '/home/Dataset/Vaihingen_1024_val'

FCRN_dir = './model/05-25/'#need to be end with .h5！
log_path = './log/05-25/'

isExists=os.path.exists(FCRN_dir)    
if not isExists:
    os.makedirs(FCRN_dir) 
isExists=os.path.exists(log_path)
if not isExists:
    os.makedirs(log_path) 

def scale_invarient_error(y_true,y_pred):
    y_p=K.clip(y_pred,K.epsilon(),np.inf)+1.#
    y_t=K.clip(y_true,K.epsilon(),np.inf)+1.#
    return K.mean(K.square(y_p-y_t),axis=-1)-Lambda*K.square(K.mean(y_p-y_t,axis=-1))
    # return K.mean(K.square(K.log(y_p)-K.log(y_t)),axis=-1)+Lambda*K.mean(abs(y_p-y_t),axis=-1) #K.square()

def gen_batch(X1, X2, batch_size):

    while True:
        idx = np.random.choice(X1.shape[0], batch_size, replace=False) #random choice
        # X1.shape[0]为所有数据总量
        yield X1[idx], X2[idx]
        
def process_line(line):  
    tmp = Img.imread(line) 
    tmp = np.array(tmp)
    width = tmp.shape[1]
    x = tmp[:,:width//2,:]  #Data
    y = tmp[:,width//2:,:]  #label,in P_V_1024 label has 3 channel！！
    return x,y  

def rescale(data):
    data=data.astype('float32')
    data /= 255.0   
    return data

def generate_arrays_from_file(input_paths,batch_size):  
    while 1:  
        random.shuffle(input_paths)# every epoch shuffle  
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
                X = np.array([cv2.pyrDown(X[i]) for i in range(len(X))])
                # X = misc.imresize(X,0.5)# input is 512
                Y = np.array(Y) 
                Y = Y[:,:,:,0] # only take one channel!
                # print Y.shape
                for i in range(1):
                    # Y = misc.imresize(Y,0.5)#output is 256
                    Y = np.array([cv2.pyrDown(Y[i]) for i in range(len(Y))])
                Y = np.expand_dims(Y,axis=-1) # output is (?,?,?,?)
                # print Y.shape
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

def identity_block_last(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1),dilation_rate=2, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,dilation_rate=2,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), dilation_rate=2,name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = keras.layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block_last(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.
    # Returns
        Output tensor for the block.
    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), dilation_rate=2,strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, dilation_rate=2,padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), dilation_rate=2,name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x =  keras.layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x

# net definition
def Up_Projection(x,f,num):

    # x = UpSampling2D(size=(2, 2))(x)
    x = Deconv2D(f, (1, 1),  strides=(2, 2), padding="same")(x)
    x1 = Conv2D(f, (5, 5), name='con5_main_'+str(num), padding="same")(x)
    x1 = Activation("relu")(x1)
    x1 = Conv2D(f, (3, 3), name='con3_main_'+str(num), padding="same")(x1)
    x2 = Conv2D(f, (5, 5), name='con5_proj_'+str(num), padding="same")(x)
    # must channel last
    x = Concatenate(axis=-1)([x1, x2])
    x = Activation("relu")(x)
    return x

def FCRN(model_name):
    # 从con层输出
    #默认参数：include_top=True, weights='imagenet',input_tensor=None, input_shape=None,
    #pooling=None, classes=1000
    # inputs=Input(shape=)
    base_model = keras.applications.resnet50.ResNet50(include_top=False,input_shape=(int(img_row/2),int(img_cols/2),3),
                                                      weights=None,pooling=None)
    #pop the last avepooling                                                  
    base_model.layers.pop()
    x = base_model.get_layer('activation_40').output
    # 最后只缩小16倍！=32
    x = conv_block_last(x, 3, [512, 512, 2048], stage=5, block='a',strides=(1,1))
    x = identity_block_last(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block_last(x, 3, [512, 512, 2048], stage=5, block='c')

    
    x = Conv2D(1024, (1, 1), name='con2D_1', padding="same")(x)
    x = BatchNormalization(axis=-1)(x)
    #1024->64
    nb_conv = [9,8,7,6]
    num = 0
    for i in nb_conv:
        num = num + 1  #for name
        x = Up_Projection(x,2**i,num)
    x = Conv2D(1, (3, 3), name='con2D_last', padding="same")(x)

    FCRN =  Model(inputs=base_model.input,outputs=x,name = model_name)
    return FCRN

def berHu(y_true,y_pred,c):
    x = abs(y_true-y_pred)
    if x<c:
        return x
    else:
        return (x**2+c**2)/(2*c)

def step_decay(epoch):
    lr = base_lr * math.pow (gamma ,math.floor(epoch / epochs_drop))
    return lr

def train():
    # Create optimizers
    opt_dcgan = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # opt_dcgan = SGD(base_lr,momentum)

    inputs_path,train_num = load_data(dset)
    val_path,val_num = load_data(dset)
    batches = generate_arrays_from_file(inputs_path,batch_size=batch_size)
    val_batches = generate_arrays_from_file(inputs_path,batch_size=batch_size)
    # callback_dis = TensorBoard('../logs/Dis/')
    # callback_dis.set_model(discriminator_model)
    # train_dis_names = ["train_loss"]
    
    FCRNmodel = FCRN('FCRN')
    tensorboard = TensorBoard(log_dir=log_path)
    # lrate = LearningRateScheduler(step_decay)
    FCRNmodel.compile(loss=scale_invarient_error,optimizer=opt_dcgan,metrics=['accuracy'])
    # progbar = generic_utils.Progbar(train_num)
    print("Start training")
    # progbar.add(batch_size, values=[("logloss", scale_invarient_error)])
    #print net info
    FCRNmodel.summary()
    FCRNmodel.fit_generator(batches,samples_per_epoch=math.ceil(train_num/batch_size) ,nb_epoch=nb_epoch,
    callbacks=[tensorboard],validation_data=val_batches,validation_steps=math.ceil(val_num/Val_batch_size))#lrate,
    FCRNmodel.save(FCRN_dir+'FCRN_predict.h5')
    return

train()


    
    




    



