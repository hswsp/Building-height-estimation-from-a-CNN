# encoding: utf-8  
""" 
 
"""  
  
import numpy as np  
from keras.models import Sequential  
from keras.layers.core import Dense, Activation  
from keras.optimizers import SGD  
from keras.utils import np_utils      
from keras.utils.vis_utils import plot_model  
  
  
def run():  
    # 构建神经网络  
    model = Sequential()  
    model.add(Dense(4, input_dim=2, init='uniform'))  
    model.add(Activation('relu'))  
    model.add(Dense(2, init='uniform'))  
    model.add(Activation('sigmoid'))  
    sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)  
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])  
  
    # 神经网络可视化  
    plot_model(model, to_file='model.png',show_shapes=True, show_layer_names=True)  
  
if __name__ == '__main__':  
    run()  
