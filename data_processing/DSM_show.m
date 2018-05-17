%此函数用于显示归一化后的灰度图，供查看
[filename,filepath] = uigetfile('*.*','Select the image');  
if isequal(filename,0)||isequal(filepath,0)
    return;
 else
    filefullpath=[filepath,filename]; %文件路径
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