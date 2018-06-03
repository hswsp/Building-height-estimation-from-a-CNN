clear;
clc;
[filename,filepath] = uigetfile('*.*','Select the image');  
if isequal(filename,0)||isequal(filepath,0)
    return;
end

fileFolder=fullfile(filepath); %打开刚刚打开图片所在的文件夹
dirOutput=dir(fullfile(fileFolder,'*.jpg'));%获取所有.tif
fileNames={dirOutput.name}'; %获得名称

%% 对应的深度图
[DSMname,DSMpath] = uigetfile('*.*','Select the DSM');  
if isequal(DSMname,0)||isequal(DSMpath,0)
    return;
end
DSMFolder=fullfile(DSMpath); %打开刚刚打开图片所在的文件夹
dirDSMOutput=dir(fullfile(DSMFolder,'*.jpg'));%获取所有.png
DSMNames={dirDSMOutput.name}'; %获得名称
if length(fileNames)~= length(DSMNames)
    disp('number of fileName do not match !');
    return;
end
paths='D:\FYP\dataset\Vaihingen_1024_merge'; %获取指定文件夹目录
system(['mkdir ',paths]);%创建保存归一化图路径
 for i= 1:length(fileNames) 
      splitname=strsplit(fileNames{i},'.'); %返回cell
      name= splitname{1}; %去除后缀           
      filefullpath=[filepath,fileNames{i}];
      Im=imread(filefullpath);
       %DSM
      DSMsplitname=strsplit(DSMNames{i},'.'); %返回cell
      % DSM_name= DSMsplitname{1}; %去除后缀
      DSMfullpath=[DSMpath,DSMNames{i}];
      DSM=imread(DSMfullpath); %读出海拔信息
      labelR(:,:,1) = DSM; 
      labelG(:,:,1) = DSM;
      labelB(:,:,1) = DSM;
      label = cat(3,labelR,labelG,labelB);
      pic = cat(2,Im,label);
      % if 8-bit jpg
      imwrite(pic,[paths,'\',name,'.jpg']);  
      %clear 
      clear label;
      clear labelR;
      clear labelG;
      clear labelB;
 end
      
      
      