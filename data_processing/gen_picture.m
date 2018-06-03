clear;
clc;
%% 全局变量
H = 1500;
W = 1500;
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
dirOutput=dir(fullfile(fileFolder,'*.tif'));%获取所有jpg.
fileNames={dirOutput.name}'; %获得名称

%% 对应的深度图
[DSMname,DSMpath] = uigetfile('*.*','Select the DSM');  
if isequal(DSMname,0)||isequal(DSMpath,0)
    return;
end

DSMFolder=fullfile(DSMpath); %打开刚刚打开图片所在的文件夹
dirOutput=dir(fullfile(DSMFolder,'*.tif'));%获取所有.jpgpng
DSMNames={dirOutput.name}'; %获得名称

if length(fileNames)~= length(DSMNames)
    disp('images number do not match !');
    return;
end

% %num of image
% index = 1; 
% images = zeros(W,H,3); %作为第一页
% depths = zeros(W,H);
%  %% if 8-bit
% images = uint8(images);
% depths = uint8(depths);
paths = 'D:\FYP\dataset\Potsdam_1500';
system(['mkdir ',paths]);%创建保存归一化图路径
%% 逐个分割
% for num = 1:3 %翻转
%     for rnum = 1:4 %旋转
%         rotate = [0,90,180,270];
    
        for i= 1:length(fileNames)   
            %filepath是文件夹 fileNames{}是各个文件名称
            splitname=strsplit(fileNames{i},'.'); %返回cell
            nameI= splitname{1}; %去除后缀
            filefullpath=[filepath,fileNames{i}];
            Im=imread(filefullpath);
%             if num == 2
%                 Im = fliplr(Im);
%             elseif num == 3
%                 Im = flipud(Im);
%             end                   
%             Im = imrotate(Im,rotate(rnum));  
            L = size(Im);
            %DSM
            DSMsplitname=strsplit(DSMNames{i},'.'); %返回cell
            DSM_name= DSMsplitname{1}; %去除后缀
            DSMfullpath=[DSMpath,DSMNames{i}];
            DSM=imread(DSMfullpath); %读出海拔信息
            %% 减去整体最小，但是应该分块减更准
%             [minI,index1]=min(DSM(:));
%             DSM=DSM-minI;

%             if num == 2
%                 DSM = fliplr(DSM);
%             elseif num == 3
%                 DSM = flipud(DSM);
%             end                   
%             DSM = imrotate(DSM,rotate(rnum));
            DSML = size(DSM);
            if DSML(1)~=L(1)
                disp('images shape do not match !');
                return;
            end
            h_val=height*(1-x);
            w_val=width*(1-x);
            max_row = (L(1)-height)/h_val+1;
            max_col = (L(2)-width)/w_val+1;

            max_row=fix(max_row);
            max_col=fix(max_col);

            for row = 1:max_row      
                for col = 1:max_col        
                     images =Im((row-1)*h_val+1:height+(row-1)*h_val,(col-1)*w_val+1:width+(col-1)*w_val,:); %其余完整部分
                     depths = DSM((row-1)*h_val+1:height+(row-1)*h_val,(col-1)*w_val+1:width+(col-1)*w_val); %其余完整部分 
                     %% 换成高度
                     [minI,index1]=min(depths(:));
                     depths=depths-minI;
%                      %% 写入tiff
%                      t = Tiff([paths,'\','images\',nameI,'_',strcat('row',num2str(row,'%02d'),'_','col',num2str(col,'%02d'),'.tif')], 'w'); 
%                      tagstruct.ImageLength     = size(images,1);  
%                      tagstruct.ImageWidth      = size(images,2);  
%                      tagstruct.SampleFormat    = Tiff.SampleFormat.IEEEFP;  
%                      tagstruct.BitsPerSample   = 32;  
%                      tagstruct.SamplesPerPixel = 3;  
%                      tagstruct.Photometric     = Tiff.Photometric.MinIsBlack;  
%                      tagstruct.RowsPerStrip    = 16;  
%                      tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;  
%                      tagstruct.Software        = 'MATLAB';  
%                      t.setTag(tagstruct)  
%                      t.write(images);  
%                      t.close();  
%                      t = Tiff([paths,'\','depths\',DSM_name,'_',strcat('row',num2str(row,'%02d'),'_','col',num2str(col,'%02d'),'.tif')], 'w');  
%                      tagstruct.ImageLength     = size(depths,1);  
%                      tagstruct.ImageWidth      = size(depths,2);  
%                      tagstruct.SampleFormat    = Tiff.SampleFormat.IEEEFP;  
%                      tagstruct.BitsPerSample   = 32;  
%                      tagstruct.SamplesPerPixel = 1;  
%                      tagstruct.Photometric     = Tiff.Photometric.MinIsBlack;  
%                      tagstruct.RowsPerStrip    = 16;  
%                      tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;  
%                      tagstruct.Software        = 'MATLAB';  
%                      t.setTag(tagstruct)  
%                      t.write(depths);
%                      t.close();
                   %% 如果有前面两层循环的话 写入.jpg
                     imwrite(images,[paths,'\','images\',nameI,'_',strcat(int2str(num),'_',int2str(rotate(rnum)),'_','row',num2str(row,'%02d'),'_','col',num2str(col,'%02d'),'.jpg')]);
                     imwrite(depths,[paths,'\','depths\',DSM_name,'_',strcat(int2str(num),'_',int2str(rotate(rnum)),'_','row',num2str(row,'%02d'),'_','col',num2str(col,'%02d'),'.jpg')]);                     
                     
                end
            end       
        end 
%     end
% end
