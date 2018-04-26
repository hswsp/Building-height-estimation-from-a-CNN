clear,clc;
[filename,filepath] = uigetfile('*.*','Select the image');  
if isequal(filename,0)||isequal(filepath,0)
    return;
% else
%    filefullpath=[filepath,filename]; %文件路径
end

fileFolder=fullfile(filepath); %打开刚刚打开图片所在的文件夹
dirOutput=dir(fullfile(fileFolder,'*.tif'));%获取所有.tf
fileNames={dirOutput.name}'; %获得名称
%逐个分割
for i=1:length(fileNames)
    
% %得到文件名，以便建立一个文件夹保存分块图像
% [pathstr,name,ext] = fileparts(filename); %该函数用于将一个文件的完整路径中各部分提取出来。

%filepath是文件夹 fileNames{}是各个文件名称
splitname=strsplit(fileNames{i},'.'); %返回cell
name= splitname{1}; %去除后缀
filefullpath=[filepath,fileNames{i}];
Im=imread(filefullpath);
% imshow(Im)
L = size(Im);
%分块大小
height=128;
width=128;
%重叠
x=0.25;
h_val=height*(1-x);
w_val=width*(1-x);
max_row = (L(1)-height)/h_val+1;
max_col = (L(2)-width)/w_val+1;
% %判断能否完整分块
% %fix向0靠拢取整
% if max_row==fix(max_row)%判断是否能够整分
%     max_row=max_row;
% else
%     max_row=fix(max_row+1);
% end
% 
% if max_col==fix(max_col)%判断是否能够整分
%     max_col=max_col;
% else
%     max_col=fix(max_col+1);
% end
%只要完整部分
max_row=fix(max_row);
max_col=fix(max_col);
seg = cell(max_row,max_col);
for row = 1:max_row      
    for col = 1:max_col        
%         if ((width+(col-1)*w_val)>L(2)&&((row-1)*h_val+height)<=L(1))%判断最右边不完整的部分
%     seg(row,col)= {Im((row-1)*h_val+1:height+(row-1)*h_val,(col-1)*w_val+1:L(2),:)};
%         elseif((height+(row-1)*h_val)>L(1)&&((col-1)*w_val+width)<=L(2))%判断最下边不完整的部分
%     seg(row,col)= {Im((row-1)*h_val+1:L(1),(col-1)*w_val+1:width+(col-1)*w_val,:)}; 
%         elseif((width+(col-1)*w_val)>L(2)&&((row-1)*h_val+height)>L(1))%判断最后一张
%     seg(row,col)={Im((row-1)*h_val+1:L(1),(col-1)*w_val+1:L(2),:)};       
%         else
     seg(row,col)= {Im((row-1)*h_val+1:height+(row-1)*h_val,(col-1)*w_val+1:width+(col-1)*w_val,:)}; %其余完整部分  
%         end
    end
end 
 imshow(Im);
 hold on
 system(['mkdir ',name]);%创建与图片名相同的文件用来保存图片
 paths=[pwd,'\',name]; %获取指定文件夹目录
 %保存子图
for i=1:max_row
    for j=1:max_col
 imwrite(seg{i,j},[paths,'\',strcat('row',int2str(i),'_','col',int2str(j),'.bmp')]);   %把第i帧的图片写为'mi.bmp'
    end
end
% 画框显示
 for row = 1:max_row      
    for col = 1:max_col  
    c=rand(1,3);%随机颜色
     rectangle('Position',[w_val*(col-1),h_val*(row-1),width,height],...
         'LineWidth',2,'LineStyle','-','EdgeColor',c);
    end
 end 
end