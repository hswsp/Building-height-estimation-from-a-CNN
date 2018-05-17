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
dirDSMOutput=dir(fullfile(DSMFolder,'*.png'));%获取所有.jpg
DSMNames={dirDSMOutput.name}'; %获得名称
if length(fileNames)~= length(DSMNames)
    disp('number of fileName do not match !');
    return;
end
paths='D:\FYP\Vaihingen'; %获取指定文件夹目录
system(['mkdir ',paths]);%创建保存归一化图路径
for i= 1:length(fileNames)        
        %filepath是文件夹 fileNames{}是各个文件名称
        splitname=strsplit(fileNames{i},'.'); %返回cell
        name= splitname{1}; %去除后缀 
        filefullpath=[filepath,fileNames{i}];
        Im=imread(filefullpath);
        %% if use 16-bit png DSM
        Im = uint16(Im);
        %%
        L = size(Im);
        %DSM
        DSMsplitname=strsplit(DSMNames{i},'.'); %返回cell
        % DSM_name= DSMsplitname{1}; %去除后缀
        DSMfullpath=[DSMpath,DSMNames{i}];
        DSM=imread(DSMfullpath); %读出海拔信息
        %% if use 16-bit png DSM
%         DSM= uint16(DSM);
        %%
        L1 = size(DSM);
        
        if L(1:2)~=L1
            disp(['picture dim not match at',fileNames{i},'!']);
        end
        %gray2rgb
        labelR =  zeros(L1(1),L1(2));
        labelB = zeros(L1(1),L1(2));
        labelG = zeros(L1(1),L1(2));
        
        labelR(:,:,1) = DSM; 
        labelG(:,:,1) = DSM;
        labelB(:,:,1) = DSM;
        label = cat(3,labelR,labelG,labelB);
        
        %分块大小
        height =1500;
        width =1500;
        %重叠
        x=0;
        h_val=height*(1-x);
        w_val=width*(1-x);
        max_row = (L(1)-height)/h_val+1;
        max_col = (L(2)-width)/w_val+1;
        max_row=fix(max_row);
        max_col=fix(max_col);
        seg = cell(max_row,max_col);
        segd = cell(max_row,max_col);
        for row = 1:max_row      
            for col = 1:max_col        
             seg(row,col)= {Im((row-1)*h_val+1:height+(row-1)*h_val,(col-1)*w_val+1:width+(col-1)*w_val,:)}; %其余完整部分  
             segd(row,col)= {label((row-1)*h_val+1:height+(row-1)*h_val,(col-1)*w_val+1:width+(col-1)*w_val,:)}; %其余完整部分  
            end
        end             
        %保存子图
        for k=1:max_row
            for j=1:max_col
                pic = cat(2,seg{k,j},segd{k,j});
                % if 8-bit jpg
%                 imwrite(pic,[paths,'\',name,'_',strcat('row',int2str(k),'_','col',int2str(j),'.jpg')]);  
                % if 16-bit png
                imwrite(pic,[paths,'\',name,'_',strcat('row',int2str(k),'_','col',int2str(j),'.png')],'bitdepth',16); 
            end
        end
end
        
       
        

