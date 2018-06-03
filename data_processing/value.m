[filename,filepath] = uigetfile('*.*','Select the output');  
if isequal(filename,0)||isequal(filepath,0)
    return;
end

outputFolder=fullfile(filepath); %�򿪸ոմ�ͼƬ���ڵ��ļ���
dirOutput=dir(fullfile(outputFolder,'*.png'));%jpg
outputNames={dirOutput.name}'; %�������

%% ��Ӧ�����ͼ
[targetname,targetpath] = uigetfile('*.*','Select the target');  
if isequal(targetname,0)||isequal(targetpath,0)
    return;
end

targetFolder=fullfile(targetpath); %�򿪸ոմ�ͼƬ���ڵ��ļ���
dirOutput=dir(fullfile(targetFolder,'*.png'));%jpg
targetNames={dirOutput.name}'; %�������

if length(outputNames)~= length(targetNames)
    disp('do not match !');
    return;
end
num = length(outputNames);
RMSE = zeros(num);
ZNCC =zeros(num);
MAE = zeros(num);
for i= 1:length(outputNames)
    splitname=strsplit(outputNames{i},'.'); %����cell
    filefullpath=[filepath,outputNames{i}];
    out=imread(filefullpath);
    out = out(:,:,1);
    out = double(out(:));
    targetsplitname=strsplit(targetNames{i},'.'); %����cell
    targetfullpath=[targetpath,targetNames{i}];
    target=imread(targetfullpath);
    target = target(:,:,1);
    target = double(target(:));
    dev_o = std(out);
    man_o = mean(out);
    dev_t = std(target);
    man_t = mean(target);
    
    num =size(out);
    num = num(1);
    RMSE(i)=sqrt(1/num*sum((target-out).^2));
    MAE(i) = 1/num*sum(abs(target-out));
    ZNCC(i)=1/num*sum(((out-man_o).*(target-man_t))./(dev_o*dev_t));
    
end

disp( mean(RMSE));
disp( mean(MAE));
disp( mean(ZNCC));