%�˺���������ʾ��һ����ĻҶ�ͼ�����鿴
[filename,filepath] = uigetfile('*.*','Select the image');  
if isequal(filename,0)||isequal(filepath,0)
    return;
 else
    filefullpath=[filepath,filename]; %�ļ�·��
end
%  origin = imread(filefullpath);
   img = imread(filefullpath);
%  row_mid = size(origin,1)/2;
%  col_mid = size(origin,2)/2;
%  row_start=row_mid-2000; row_end=row_mid+2000;
%  col_start=col_mid-2000;col_end=col_mid+2000;
%  
%  img=origin(row_start:row_end,col_start:col_end);
%  img = img-min(img(:));
%  img = img*255/(max(img(:)));
%  img = uint8(img);
 myLogEnhance(img,10);
 img = histeq(img);
 imshow(img);