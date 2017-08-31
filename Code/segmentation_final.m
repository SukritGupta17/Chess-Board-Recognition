close all;
clear all;
load('int1.mat');

figure;
imshow(sg_img);
sg_img_g = rgb2gray(sg_img);
BW = imbinarize(sg_img_g,'adaptive','ForegroundPolarity','dark','Sensitivity',0.4);
figure;
imshow(BW);

se = strel('disk',7);
se1 = strel('line',6,60);
afterOpening = imopen(BW,se);
afterOpening = imdilate(afterOpening,se);
figure
I = imread('chess8.jpg');
I = imresize(I, 0.5);
figure;
imshow(I);
I_g = rgb2gray(I);
BW = edge(I_g,'sobel', 0.4);
imshow(BW);

% imcontour(afterOpening,2);
stats = regionprops('table',afterOpening,'BoundingBox');
bb = stats.BoundingBox;

for i=1:size(bb,1)
    bb(i,1) = bb(i,1)-50;
    bb(i,2) = bb(i,2)-50
    bb(i,3) = bb(i,3)+75;
    bb(i,4) = bb(i,4)+75;  
end

for i=1:size(bb,1)
rectangle('Position', bb(i,:),'EdgeColor','r', 'LineWidth', 1);
end
for i=1:size(bb,1)
rectangle('Position', bb(i,:),'EdgeColor','r', 'LineWidth', 1);
I2 = imcrop(I,bb(i,:));

filename = (['new_frame_',num2str(i),'.jpg']);
imwrite( I2, filename );
end


    