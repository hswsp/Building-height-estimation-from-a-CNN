function dst_img=myLogEnhance(src_img,v)   
c=1.0;  
src_img = mat2gray(src_img,[0 255]);  
g =c*log2(1 + v*src_img)/log2(v+1);  
%∑¥πÈ“ªªØ  
max=255;  
min=0;  
dst_img=uint8(g*(max-min)+min);  

