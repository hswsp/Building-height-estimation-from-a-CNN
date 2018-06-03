# Height Simulation From Single Image

This is for building height estimation from a single image using convolutional neural networks

*data_processing* is for preparing dataset for each network

*largeImage* uses the net in [1], but it seem not suitable for height estimation

*Multi_scale* uses the net in [2], it is a small but useful net for this task

*FCRN* uses the net in [3], although it is very usefull, it takes more than 15 times than *Multi_sclae* to trian. I only tried the L2 loss and it causes blur.

*pix2pix* uses the cGAN in [4]，which is the most usefull net for solving this problem with great potential.

## Reference
[1]Xiaowei. Zhou, Guoqiang. Zhong, Lin. Qi, Junyu. Dong,Tuan D. Pham,  Jianzhou. Mao, Surface Height Map Estimation from a Single Image Using Convolutional Neural Networks, Eighth International Conference on Graphic and Image Processing (ICGIP), 2016, 1022524

[2]David. Eigen, Christian. Puhrsch, Rob. Fergus, Depth Map Prediction from a Single Image using a Multi-Scale Deep Network, NIPS, 2014, 5539, pp 2366 - 2374

[3]Iro. Laina, Christian. Rupprecht, Vasileios. Belagiannis, Deeper Depth Prediction with Fully Convolutional Residual Networks, IEEE International Conference on 3D Vision, 2016, pp 239 - 248

[4]Phillip. Isola, Jun-Yan. Zhu, Tinghui. Zhou, Alexei A. Efros, Image-to-image translation with conditional adversarial networks, CVPR, 2017, pp 5967 – 5976
