close all;
clear all;
f=imread('chess2.jpg');
f = rgb2gray(f);
pout_adapthisteq = histeq(f);
figure;
imshow(pout_adapthisteq);
f = pout_adapthisteq;

f=im2double(f);
%global Thresholding
T=0.5*(min(f(:))+max(f(:)));
done=false;
while ~done
    g=f>=T;
    Tn=0.5*(mean(f(g))+mean(f(~g)));
    done=abs(T-Tn)<0.1;
    T=Tn;
end
display('Threshold(T) - Iterative');
T
r=im2bw(f,T);
% figure,imshow(f),title('Original Image');
% figure,imshow(r),title('Global Thresholding - Iterative Method');
Th=graythresh(f);
display('Threshold(T) - Otsu''s Method');
Th
s=im2bw(f,Th);
% figure,imshow(s),title('Global Thresholding - Otsu''s Method');
%Local Thresholding
se=strel('sphere',90);
ft=imtophat(f,se);
Thr=graythresh(ft);
display('Threshold(T) - Local Thresholding');
Thr
lt=im2bw(ft,Thr);
figure,imshow(lt),title('Local Thresholding');