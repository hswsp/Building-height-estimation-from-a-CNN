clc;
clear;
ParentFolder='D:\Github_repository\FYP_dataset\dataset\';
ParentPath = dir(ParentFolder);

Parent_data_Folder=[ParentFolder,ParentPath(3).name]; %第一个为数据集
Parent_data_Path = dir(Parent_data_Folder);
Num_data_Folders = length(Parent_data_Path);

Parent_label_Folder=[ParentFolder,ParentPath(4).name]; %第二个为标注集
Parent_label_Path = dir(Parent_label_Folder);
Num_label_Folders = length(Parent_label_Path);

if Num_data_Folders~=Num_label_Folders
    disp(['label and data do not match!']);
else
    NumFolders=Num_data_Folders;
end

TotalTrainNum=0;
TotalTestNum=0;
 
for i = 3:floor(0.8*NumFolders)
     %data
    Folder_data_Path = [Parent_data_Folder,'\',Parent_data_Path(i).name];  %依次进入data下每一个文件,与label对应
    imageName=dir(Folder_data_Path);    
    numPic=length(imageName);
    TotalTrainNum=TotalTrainNum+(numPic-2)
end
for i = (floor(0.8*NumFolders)+1):NumFolders
     %data
    Folder_data_Path = [Parent_data_Folder,'\',Parent_data_Path(i).name];  %依次进入data下每一个文件,与label对应
    imageName=dir(Folder_data_Path);    
    numPic=length(imageName);
    TotalTestNum=TotalTestNum+(numPic-2)
end

 disp(['训练样本数',num2str(TotalTrainNum)]);
 disp(['测试样本数',num2str(TotalTestNum)]);
