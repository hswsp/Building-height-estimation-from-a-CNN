%% 用于归一化DSM,38.3015->38.30,大于655.35的一律为65535
%% 对于使用8-bit，大于25.5一律显示255
[filename,filepath] = uigetfile('*.*','Select the image');  
if isequal(filename,0)||isequal(filepath,0)
    return;
end
fileFolder=fullfile(filepath); %打开刚刚打开图片所在的文件夹
dirOutput=dir(fullfile(fileFolder,'*.tif'));%获取所有.tf
fileNames={dirOutput.name}'; %获得名称
paths = 'D:\FYP\DSM_Vaihingen_norm_8bit';
system(['mkdir ',paths]);%创建保存归一化图路径
%% 变成高度，存储为16位图
for i= 1:length(fileNames)        
        %filepath是文件夹 fileNames{}是各个文件名称
        splitname=strsplit(fileNames{i},'.'); %返回cell
        name= splitname{1}; %去除后缀 
        filefullpath=[filepath,fileNames{i}];
        Im=imread(filefullpath);
       %% if 8-bits
        Im = Im * 10;
        %变成高度信息
        [minI,index]=min(Im(:));
        Im=Im-minI;
        L = size(Im);
        %% if 16-bits
%         Im = Im * 100;
%         Im = uint16(Im);
%         imwrite(Im,[paths,'\',name,'.png'],'png','bitdepth',16);
        Im = Im*255.0/(max(Im(:)));
%         for sight
        Im = uint8(Im);
%         myLogEnhance(Im,10);
%         Im = histeq(Im);
        imwrite(Im,[paths,'\',name,'.jpg']);
end
        