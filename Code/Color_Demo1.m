close all;
clear all;
% Read in image
I = imread('chess8.jpg');

% Scale down 50%
I = imresize(I, 0.5);

% Convert to grayscale
Ig = rgb2gray(I);

% Get a gaussian kernel for blurring
K = fspecial('gaussian',[10 10],3);

% Blur the image
% Note that multiple passes with a fixed kernel
% are the same as a single pass with a larger kernel
Igf = imfilter(Ig, K);
Igf = imfilter(Igf, K);
Igf = imfilter(Igf, K);

[counts,x] = imhist(Igf,10);
stem(x,counts);
T = otsuthresh(counts);
BW = imbinarize(Igf,T);
f1 = figure;
subplot(1,2,1);
imshow(Igf);
title('Gaussian Filtered Image');
subplot(1,2,2);
imshow(BW);
title('Otsu Thresholded Image');

% Otsu is better if not change to gray thresholding
% figure;
% 
% level = graythresh(Igf)
% BW1 = imbinarize(Igf,level);
% imshow(BW1);

% II = Igf - (255*uint8(BW));
% II = uint8(II);
% 
% for i=1:size(II,1)
%     for j=1:size(II,2)
%         if II(i,j)==0
%             I(i,j,1)=0;
%             I(i,j,2)=0;
%             I(i,j,3)=0;
%         end
%     end
% end
%color based clustering

HSV = rgb2hsv(I);
h = HSV(:,:,1);
s = HSV(:,:,2);
v = HSV(:,:,3);
f2 = figure;
subplot(1,3,1);
imshow(h);
subplot(1,3,2);
imshow(s);
subplot(1,3,3);
imshow(v);

cform = makecform('srgb2lab');
lab_he = applycform(I,cform);

l = lab_he(:,:,1);
a = lab_he(:,:,2);
b = lab_he(:,:,3);
f3 = figure;
subplot(1,3,1);
imshow(l);
subplot(1,3,2);
imshow(a);
subplot(1,3,3);
imshow(b);

ab = double(s);
nrows = size(ab,1);
ncols = size(ab,2);
ab = reshape(ab,nrows*ncols,1);

nColors = 2;
% repeat the clustering 3 times to avoid local minima
[cluster_idx, cluster_center] = kmeans(ab,nColors,'distance','sqeuclidean', ...
                                      'Replicates',10);


pixel_labels = reshape(cluster_idx,nrows,ncols);
% imshow(pixel_labels,[]), title('image labeled by cluster index');
segmented_images = cell(1,2);
rgb_label = repmat(pixel_labels,[1 1 3]);

for k = 1:nColors
    color = I;
    color(rgb_label ~= k) = 0;
    segmented_images{k} = color;
end

f4 = figure;imshow(segmented_images{1}), title('objects in cluster 1');
f5 = figure;imshow(segmented_images{2}), title('objects in cluster 2');

I = imread('chess8.jpg');
I = rgb2gray(I);
I = imresize(I, 0.5);
% Get a gaussian kernel for blurring
K = fspecial('gaussian',[10 10],0.3);

% Blur the image
% Note that multiple passes with a fixed kerne5
% are the same as a single pass with a larger kernel
I = imfilter(I, K);
I = imfilter(I, K);
I = imfilter(I, K);
I = medfilt2(I);
% figure, imshow(I), title('original image');

BWs = I;
[~, threshold] = edge(I, 'sobel');
fudgeFactor = .6;
BWs = edge(I,'sobel', threshold * fudgeFactor);

se90 = strel('line', 5, 45);
se0 = strel('line', 5, 90);

BWsdil = imdilate(BWs, [se90 se0]);

BWdfill = imfill(BWsdil, 'holes');

BWnobord = imclearborder(BWdfill, 4);

seD = strel('diamond',1);
BWfinal = imerode(BWnobord,seD);
BWfinal = imerode(BWfinal,seD);
f6 = figure, imshow(BWfinal), title('Segmented image');
BWfinal = BWfinal*255;
if(cluster_center(2,1)<0.1)
    sg_img = segmented_images{2};
else
    sg_img = segmented_images{1};
end
for i=1:size(I,1)
    for j=1:size(I,2)
        if BWfinal(i,j)==0
            sg_img(i,j,1)=0;
            sg_img(i,j,2)=0;
            sg_img(i,j,3)=0;
        end
    end
end

sg_img_g = rgb2gray(sg_img);
BW = imbinarize(sg_img_g,'adaptive','ForegroundPolarity','dark','Sensitivity',0.4);
f7 = figure;
imshow(BW);
title('Foreground Polarity Binarized Image');

se = strel('disk',4);
afterOpening = imopen(BW,se);

afterOpening = imerode(afterOpening,se);
f8 = figure;
imshow(afterOpening,[]);

sg_img_g = rgb2gray(sg_img);
BW = imbinarize(sg_img_g,'adaptive','ForegroundPolarity','dark','Sensitivity',0.4);

se = strel('disk',7);
se1 = strel('line',6,60);
afterOpening = imopen(BW,se);
afterOpening = imdilate(afterOpening,se);
f9 = figure;
I = imread('chess8.jpg');
I = imresize(I, 0.5);

I_g = rgb2gray(I);
BW = edge(I_g,'sobel', 0.4);
f10 = figure;
imshow(I);
hold on

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
title('Detected Bounding Box');
