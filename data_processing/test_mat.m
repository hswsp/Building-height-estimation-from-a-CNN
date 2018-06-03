clear;
%用来测试.mat是否正确
load('D:\FYP\dataset\Potsdam_1024.mat');
img = images(:,:,:,340);
dep = depths(:,:,340);
figure();
subplot(1,2,1);
subplot(1,2,1);
imshow(img);
subplot(1,2,2);
imshow(dep);