clear;
clc;
[filename,filepath] = uigetfile('*.*','Select the image');  
if isequal(filename,0)||isequal(filepath,0)
    return;
end

fileFolder=fullfile(filepath); %�򿪸ոմ�ͼƬ���ڵ��ļ���
dirOutput=dir(fullfile(fileFolder,'*.jpg'));%��ȡ����.tif
fileNames={dirOutput.name}'; %�������

%% ��Ӧ�����ͼ
[DSMname,DSMpath] = uigetfile('*.*','Select the DSM');  
if isequal(DSMname,0)||isequal(DSMpath,0)
    return;
end
DSMFolder=fullfile(DSMpath); %�򿪸ոմ�ͼƬ���ڵ��ļ���
dirDSMOutput=dir(fullfile(DSMFolder,'*.jpg'));%��ȡ����.png
DSMNames={dirDSMOutput.name}'; %�������
if length(fileNames)~= length(DSMNames)
    disp('number of fileName do not match !');
    return;
end
paths='D:\FYP\dataset\Vaihingen_1024_merge'; %��ȡָ���ļ���Ŀ¼
system(['mkdir ',paths]);%���������һ��ͼ·��
 for i= 1:length(fileNames) 
      splitname=strsplit(fileNames{i},'.'); %����cell
      name= splitname{1}; %ȥ����׺           
      filefullpath=[filepath,fileNames{i}];
      Im=imread(filefullpath);
       %DSM
      DSMsplitname=strsplit(DSMNames{i},'.'); %����cell
      % DSM_name= DSMsplitname{1}; %ȥ����׺
      DSMfullpath=[DSMpath,DSMNames{i}];
      DSM=imread(DSMfullpath); %����������Ϣ
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
      
      
      