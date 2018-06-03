clear;
clc;
%% ȫ�ֱ���
H = 1500;
W = 1500;
%�ֿ��С
height=H;
width=W;
%�ص�����
x=0.0;

%% ͼƬ
%% ���ļ���
[filename,filepath] = uigetfile('*.*','Select the image');  
if isequal(filename,0)||isequal(filepath,0)
    return;
end

fileFolder=fullfile(filepath); %�򿪸ոմ�ͼƬ���ڵ��ļ���
dirOutput=dir(fullfile(fileFolder,'*.tif'));%��ȡ����jpg.
fileNames={dirOutput.name}'; %�������

%% ��Ӧ�����ͼ
[DSMname,DSMpath] = uigetfile('*.*','Select the DSM');  
if isequal(DSMname,0)||isequal(DSMpath,0)
    return;
end

DSMFolder=fullfile(DSMpath); %�򿪸ոմ�ͼƬ���ڵ��ļ���
dirOutput=dir(fullfile(DSMFolder,'*.tif'));%��ȡ����.jpgpng
DSMNames={dirOutput.name}'; %�������

if length(fileNames)~= length(DSMNames)
    disp('images number do not match !');
    return;
end

% %num of image
% index = 1; 
% images = zeros(W,H,3); %��Ϊ��һҳ
% depths = zeros(W,H);
%  %% if 8-bit
% images = uint8(images);
% depths = uint8(depths);
paths = 'D:\FYP\dataset\Potsdam_1500';
system(['mkdir ',paths]);%���������һ��ͼ·��
%% ����ָ�
% for num = 1:3 %��ת
%     for rnum = 1:4 %��ת
%         rotate = [0,90,180,270];
    
        for i= 1:length(fileNames)   
            %filepath���ļ��� fileNames{}�Ǹ����ļ�����
            splitname=strsplit(fileNames{i},'.'); %����cell
            nameI= splitname{1}; %ȥ����׺
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
            DSMsplitname=strsplit(DSMNames{i},'.'); %����cell
            DSM_name= DSMsplitname{1}; %ȥ����׺
            DSMfullpath=[DSMpath,DSMNames{i}];
            DSM=imread(DSMfullpath); %����������Ϣ
            %% ��ȥ������С������Ӧ�÷ֿ����׼
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
                     images =Im((row-1)*h_val+1:height+(row-1)*h_val,(col-1)*w_val+1:width+(col-1)*w_val,:); %������������
                     depths = DSM((row-1)*h_val+1:height+(row-1)*h_val,(col-1)*w_val+1:width+(col-1)*w_val); %������������ 
                     %% ���ɸ߶�
                     [minI,index1]=min(depths(:));
                     depths=depths-minI;
%                      %% д��tiff
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
                   %% �����ǰ������ѭ���Ļ� д��.jpg
                     imwrite(images,[paths,'\','images\',nameI,'_',strcat(int2str(num),'_',int2str(rotate(rnum)),'_','row',num2str(row,'%02d'),'_','col',num2str(col,'%02d'),'.jpg')]);
                     imwrite(depths,[paths,'\','depths\',DSM_name,'_',strcat(int2str(num),'_',int2str(rotate(rnum)),'_','row',num2str(row,'%02d'),'_','col',num2str(col,'%02d'),'.jpg')]);                     
                     
                end
            end       
        end 
%     end
% end
