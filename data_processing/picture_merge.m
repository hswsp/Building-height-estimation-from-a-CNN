[filename,filepath] = uigetfile('*.*','Select the image');  
if isequal(filename,0)||isequal(filepath,0)
    return;
end

fileFolder=fullfile(filepath); %�򿪸ոմ�ͼƬ���ڵ��ļ���
dirOutput=dir(fullfile(fileFolder,'*.jpg'));%��ȡ����.tf
fileNames={dirOutput.name}'; %�������

%% ��Ӧ�����ͼ
[DSMname,DSMpath] = uigetfile('*.*','Select the DSM');  
if isequal(DSMname,0)||isequal(DSMpath,0)
    return;
end
DSMFolder=fullfile(DSMpath); %�򿪸ոմ�ͼƬ���ڵ��ļ���
dirOutput=dir(fullfile(DSMFolder,'*.jpg'));%��ȡ����.tf-*.tif
DSMNames={dirOutput.name}'; %�������
if length(fileNames)~= length(DSMNames)
    disp('do not match !');
    return;
end

for i= 1:length(fileNames)        
        %filepath���ļ��� fileNames{}�Ǹ����ļ�����
        splitname=strsplit(fileNames{i},'.'); %����cell
        name= splitname{1}; %ȥ����׺
        
        filefullpath=[filepath,fileNames{i}];
        Im=imread(filefullpath);
        L = size(Im);
        %DSM
        DSMsplitname=strsplit(DSMNames{i},'.'); %����cell
        % DSM_name= DSMsplitname{1}; %ȥ����׺
        DSMfullpath=[DSMpath,DSMNames{i}];
        DSM=imread(DSMfullpath); %����������Ϣ
        L1 = size(DSM);
        if L(1:2)~=L1
            disp(['picture dim not match at',fileNames{i},'!']);
        end
        
        for j=1:3
            label(:,:,j)=DSM;
        end
        %�ֿ��С
        height =1500;
        width =1500;
        %�ص�
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
             seg(row,col)= {Im((row-1)*h_val+1:height+(row-1)*h_val,(col-1)*w_val+1:width+(col-1)*w_val,:)}; %������������  
             segd(row,col)= {label((row-1)*h_val+1:height+(row-1)*h_val,(col-1)*w_val+1:width+(col-1)*w_val,:)}; %������������  
            end
        end 
        
        paths='D:\FYP\Potsdam'; %��ȡָ���ļ���Ŀ¼
        %������ͼ
        for k=1:max_row
            for j=1:max_col
                pic = cat(2,seg{k,j},segd{k,j});
                imwrite(pic,[paths,'\',name,'_',strcat('row',int2str(k),'_','col',int2str(j),'.jpg')]);   %�ѵ�i֡��ͼƬдΪ'mi.bmp'
            end
        end
end
        
       
        

