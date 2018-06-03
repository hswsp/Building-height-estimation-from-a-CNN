%% ���ڹ�һ��DSM,38.3015->38.30,����655.35��һ��Ϊ65535
%% ����ʹ��8-bit������25.5һ����ʾ255
[filename,filepath] = uigetfile('*.*','Select the image');  
if isequal(filename,0)||isequal(filepath,0)
    return;
end
fileFolder=fullfile(filepath); %�򿪸ոմ�ͼƬ���ڵ��ļ���
dirOutput=dir(fullfile(fileFolder,'*.tif'));%��ȡ����.tf
fileNames={dirOutput.name}'; %�������
paths = 'D:\FYP\DSM_Vaihingen_norm_8bit';
system(['mkdir ',paths]);%���������һ��ͼ·��
%% ��ɸ߶ȣ��洢Ϊ16λͼ
for i= 1:length(fileNames)        
        %filepath���ļ��� fileNames{}�Ǹ����ļ�����
        splitname=strsplit(fileNames{i},'.'); %����cell
        name= splitname{1}; %ȥ����׺ 
        filefullpath=[filepath,fileNames{i}];
        Im=imread(filefullpath);
       %% if 8-bits
        Im = Im * 10;
        %��ɸ߶���Ϣ
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
        