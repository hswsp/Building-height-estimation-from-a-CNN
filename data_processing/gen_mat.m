clear;
clc;
%% 全局变量
H = 1024;
W = 1024;
%分块大小
height=H;
width=W;
%重叠比例
x=0.0;

%% 图片
%% 打开文件夹
[filename,filepath] = uigetfile('*.*','Select the image');  
if isequal(filename,0)||isequal(filepath,0)
    return;
end

fileFolder=fullfile(filepath); %打开刚刚打开图片所在的文件夹
dirOutput=dir(fullfile(fileFolder,'*.tif'));%获取所有.jpg
fileNames={dirOutput.name}'; %获得名称

%% 对应的深度图
[DSMname,DSMpath] = uigetfile('*.*','Select the DSM');  
if isequal(DSMname,0)||isequal(DSMpath,0)
    return;
end

DSMFolder=fullfile(DSMpath); %打开刚刚打开图片所在的文件夹
dirOutput=dir(fullfile(DSMFolder,'*.jpg'));%获取所有.png
DSMNames={dirOutput.name}'; %获得名称

if length(fileNames)~= length(DSMNames)
    disp('do not match !');
    return;
end

%num of image
index = 1; 
images = zeros(W,H,3); %作为第一页
depths = zeros(W,H);
 %% if 8-bit
images = uint8(images);
depths = uint8(depths);

%% 逐个分割
for num = 1:1
    rotate = [0,90,180,270];
%     start = floor(length(fileNames)/4)+1;
%     stop = floor(length(fileNames)/2)+1;    
    for i= 1:length(fileNames)%30 1       
        % name for variable
%         Matname =['D://FYP//dataset//Postdam','_',num2str(rotate(num),'%03d'),'_',num2str(i,'%02d')];
%         dname = ['depths','_',num2str(rotate(num),'%03d'),'_',num2str(i,'%02d')];
%         Var = struct(iname,{images},dname,{depths});
%         Var.(iname) = zeros(W,H,3); %作为第一项
%         Var.(dname) = zeros(W,H);
%         eval([iname '= images']);
%         eval([dname '= depths']);
        %filepath是文件夹 fileNames{}是各个文件名称
        splitname=strsplit(fileNames{i},'.'); %返回cell
        % name= splitname{1}; %去除后缀
        filefullpath=[filepath,fileNames{i}];
        Im=imread(filefullpath);
        Im = imrotate(Im,rotate(num));  
        L = size(Im);
%         imshow(Im);
        %DSM
        DSMsplitname=strsplit(DSMNames{i},'.'); %返回cell
        % DSM_name= DSMsplitname{1}; %去除后缀
        DSMfullpath=[DSMpath,DSMNames{i}];
        DSM=imread(DSMfullpath); %读出海拔信息
        DSM = imrotate(DSM,rotate(num));       
%         %变成高度信息
%         [minI,index1]=min(DSM(:));
%         DSM=DSM-minI;
%         DSML = size(DSM);
%         if DSML(1)~=L(1)
%             disp('do not match !');
%             return;
%         end
        h_val=height*(1-x);
        w_val=width*(1-x);
        max_row = (L(1)-height)/h_val+1;
        max_col = (L(2)-width)/w_val+1;
        % if floor(num/4)==1
        %     %翻转
        %     Im = fliplr(Im);
        %     DSM = fliplr(DSM);
        % end      
        %只要完整部分
        max_row=fix(max_row);
        max_col=fix(max_col);
        % seg = cell(max_row,max_col);
        % segDSM = cell(max_row,max_col);      
        for row = 1:max_row      
            for col = 1:max_col        
                 images(:,:,:,index) =Im((row-1)*h_val+1:height+(row-1)*h_val,(col-1)*w_val+1:width+(col-1)*w_val,:); %其余完整部分
                 depths(:,:,index) = DSM((row-1)*h_val+1:height+(row-1)*h_val,(col-1)*w_val+1:width+(col-1)*w_val); %其余完整部分 
                 index =index+1;
            end
        end       
%         if i == start
%             save('D://FYP//dataset//Postdam1', 'images','depths','-v7.3');
%         else
%             save('D://FYP//dataset//Postdam1','images','depths', '-append');
%         end
    end
       
end
%% 保存
save('D://FYP//dataset//Vaihingen_1024','images','depths','-v7.3');
