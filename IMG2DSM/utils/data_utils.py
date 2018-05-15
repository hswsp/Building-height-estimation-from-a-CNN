# -*- coding: UTF-8 -*- 
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
import h5py

import matplotlib.pylab as plt

TrainData1='Postdam.mat'
TrainData2='Postdam1.mat'
ValData = 'Postdam2.mat'               #'Vaihingen.mat'

def normalization(X):
    #[0,255]=>[-1,1]
    return X / 127.5 - 1

def normalization_float(X,maxX):
    #[0,max]=>[-1,1]
    X = X / maxX
    return 2.0*X-1

def inverse_normalization(X):
    # [-1,1]=>[0,1]
    return (X + 1.) / 2.


def get_nb_patch(img_dim, patch_size, image_data_format):

    assert image_data_format in ["channels_first", "channels_last"], "Bad image_data_format"
    if image_data_format == "channels_first":
        assert img_dim[1] % patch_size[0] == 0, "patch_size does not divide height"
        assert img_dim[2] % patch_size[1] == 0, "patch_size does not divide width"
        nb_patch = (img_dim[1] // patch_size[0]) * (img_dim[2] // patch_size[1])
        img_dim_disc = (img_dim[0], patch_size[0], patch_size[1])

    elif image_data_format == "channels_last":
        print img_dim[0]
        print patch_size[0]
        assert img_dim[0] % patch_size[0] == 0, "patch_size does not divide height"
        assert img_dim[1] % patch_size[1] == 0, "patch_size does not divide width"
        nb_patch = (img_dim[0] // patch_size[0]) * (img_dim[1] // patch_size[1])
        img_dim_disc = (patch_size[0], patch_size[1], img_dim[-1])

    return nb_patch, img_dim_disc


def extract_patches(X, image_data_format, patch_size):  #从大图中抽取patch

    # Now extract patches form X_disc
    if image_data_format == "channels_first":
        X = X.transpose(0,2,3,1)

    list_X = []
    list_row_idx = [(i * patch_size[0], (i + 1) * patch_size[0]) for i in range(X.shape[1] // patch_size[0])]
    list_col_idx = [(i * patch_size[1], (i + 1) * patch_size[1]) for i in range(X.shape[2] // patch_size[1])]

    for row_idx in list_row_idx:
        for col_idx in list_col_idx:
            list_X.append(X[:, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :])

    if image_data_format == "channels_first":
        for i in range(len(list_X)):
            list_X[i] = list_X[i].transpose(0,3,1,2)

    return list_X

def load_largeData(X_full,img_type):
    dsm_num = len(X_full)
    print dsm_num
    if img_type == 'depths':
        X_full1 = np.array(X_full[:dsm_num/4]).astype(np.float16)
        X_full2 = np.array(X_full[dsm_num / 4:dsm_num / 2]).astype(np.float16)
        X1 = np.concatenate((X_full1,X_full2), axis = 0)
        del X_full1,X_full2
        X_full3 = np.array(X_full[dsm_num / 2:3 * dsm_num / 4]).astype(np.float16)
        X_full4 = np.array(X_full[3 * dsm_num / 4:]).astype(np.float16)
        X2 = np.concatenate((X_full3, X_full4), axis=0)
        del X_full3, X_full4
        y_data = np.concatenate((X1, X2), axis=0)
    elif img_type == 'images':
        X_full1 = np.array(X_full[:dsm_num / 4])
        X_full2 = np.array(X_full[dsm_num / 4:dsm_num / 2])
        X1 = normalization(np.concatenate((X_full1.astype(np.float16), X_full2.astype(np.float16)), axis=0))
        del X_full1, X_full2
        X_full3 = np.array(X_full[dsm_num / 2:3 * dsm_num / 4])
        X_full4 = np.array(X_full[3 * dsm_num / 4:])
        X2 = normalization(np.concatenate((X_full3.astype(np.float16), X_full4.astype(np.float16)), axis=0))
        del X_full3, X_full4
        y_data = np.concatenate((X1, X2), axis=0)
    return y_data


def load_data(dset, image_data_format):

    # output:[-1,1]

    with h5py.File(dset+'test.mat', "r") as hf: #TrainData1
        y_data = hf["depths"] # 关键：这里的h5f与dataset并不包含真正的数据，只是包含了数据的相关信息，不会占据内存空间
        y_data = load_largeData(y_data,'depths')
        X_data = hf["images"]
        X_data = load_largeData(X_data,'images')
        
        hf.close()
    with h5py.File(dset+TrainData2, "r") as hm: #
        y_data1 = hm["depths"]
        y_data1 = load_largeData(y_data1,'depths')
        X_data1 = hm["images"]
        X_data1 = load_largeData(X_data1,'images')
        hm.close()

    y_data = np.concatenate((y_data, y_data1),axis = 0)
    X_data = np.concatenate((X_data, X_data1),axis = 0)

    if image_data_format == "channels_last":
        y_data = np.expand_dims(y_data, axis=3)
        y_data = normalization(y_data) # =>[-1,1] _float ,np.max(y_data)
        y_data = np.concatenate((y_data,y_data,y_data), axis = 3) # zero aixis is number, 3 is channel
        X_data = X_data.transpose(0, 2, 3, 1)
        


    with h5py.File(dset+'test_val.mat', "r") as hv:#ValData
            y_data_val = hv["depths"]
            dsm_num = len(y_data_val)
            y_data_val = np.array(y_data_val[:dsm_num / 4]).astype(np.float16)
            
            X_data_val =hv["images"]
            img_num = len(X_data_val)
            X_data_val = np.array(X_data_val[:img_num / 4]).astype(np.uint8)
           
            if image_data_format == "channels_last":
                y_data_val = np.expand_dims(y_data_val, axis=3)
                y_data_val = normalization(y_data_val)  #=>[-1,1] _float ,np.max(y_data_val)
                y_data_val = np.concatenate((y_data_val, y_data_val,y_data_val),  axis=3)
                X_data_val = normalization(X_data_val.transpose(0, 2, 3, 1))
                

    return y_data, X_data, y_data_val, X_data_val


def gen_batch(X1, X2, batch_size):

    while True:
        idx = np.random.choice(X1.shape[0], batch_size, replace=False) #random choice
        # X1.shape[0]为所有数据总量
        yield X1[idx], X2[idx]


def get_disc_batch(X_full_batch, X_sketch_batch, generator_model, batch_counter, patch_size,
                   image_data_format, label_smoothing=False, label_flipping=0):

    # Create X_disc: alternatively only generated or real images
    if batch_counter % 2 == 0:
        # Produce an output
        X_disc = generator_model.predict(X_sketch_batch) # 根据输入图片预测的图片
        y_disc = np.zeros((X_disc.shape[0], 2), dtype=np.uint8)
        y_disc[:, 0] = 1 # y_disc[:,0] represent gen image;y_disc[:,1] represent true image

        if label_flipping > 0:
            p = np.random.binomial(1, label_flipping)
            if p > 0:
                y_disc[:, [0, 1]] = y_disc[:, [1, 0]]

    else:
        X_disc = X_full_batch  #输入的标准图片
        y_disc = np.zeros((X_disc.shape[0], 2), dtype=np.uint8)
        if label_smoothing:
            y_disc[:, 1] = np.random.uniform(low=0.9, high=1, size=y_disc.shape[0])
        else:
            y_disc[:, 1] = 1

        if label_flipping > 0:
            p = np.random.binomial(1, label_flipping)
            if p > 0:
                y_disc[:, [0, 1]] = y_disc[:, [1, 0]]

    # Now extract patches form X_disc
    X_disc = extract_patches(X_disc, image_data_format, patch_size)

    return X_disc, y_disc


def plot_generated_batch(X_full, X_sketch, generator_model, batch_size, image_data_format, suffix):

    # Generate images
    X_gen = generator_model.predict(X_sketch)

    X_sketch = inverse_normalization(X_sketch)
    # X_full = inverse_normalization(X_full)
    X_gen = inverse_normalization(X_gen)


    #take the first 8 picture
    Xs = X_sketch[:8]
    Xg = X_gen[:8]
    Xr = X_full[:8]

    if image_data_format == "channels_last":
        X = np.concatenate((Xs, Xg, Xr), axis=0)
        list_rows = []
        for i in range(int(X.shape[0] // 4)): # = 6
            Xr = np.concatenate([X[k] for k in range(4 * i, 4 * (i + 1))], axis=1) # 按行拼起来（上下）
            list_rows.append(Xr)

        Xr = np.concatenate(list_rows, axis=0)

    if image_data_format == "channels_first":
        X = np.concatenate((Xs, Xg, Xr), axis=0)
        list_rows = []
        for i in range(int(X.shape[0] // 4)):
            Xr = np.concatenate([X[k] for k in range(4 * i, 4 * (i + 1))], axis=2)
            list_rows.append(Xr)

        Xr = np.concatenate(list_rows, axis=1)
        Xr = Xr.transpose(1,2,0)

    if Xr.shape[-1] == 1:
        plt.imshow(Xr[:, :, 0], cmap="gray")
    else:
        plt.imshow(Xr)
    plt.axis("off")
    plt.savefig("../figures/current_batch_%s.png" % suffix)
    plt.clf()
    plt.close()
