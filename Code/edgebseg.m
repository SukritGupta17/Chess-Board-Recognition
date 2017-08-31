%% Step 1: Read Image
% Read in the |cell.tif| image, which is an image of a prostate cancer
% cell.

close all;
clear all;
I = imread('chess8.JPG');
I = rgb2gray(I);

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
% figure, imshow(BWs), title('binary gradient mask');

%% Step 3: Dilate the Image
% The binary gradient mask shows lines of high contrast in the image. These
% lines do not quite delineate the outline of the object of interest.
% Compared to the original image, you can see gaps in the lines surrounding
% the object in the gradient mask. These linear gaps will disappear if the
% Sobel image is dilated using linear structuring elements, which we can
% create with the |strel| function.

se90 = strel('line', 5, 45);
se0 = strel('line', 5, 90);

%%
% The binary gradient mask is dilated using the vertical structuring
% element followed by the horizontal structuring element. The |imdilate|
% function dilates the image.

BWsdil = imdilate(BWs, [se90 se0]);
% figure, imshow(BWsdil), title('dilated gradient mask');

%% Step 4: Fill Interior Gaps 
% The dilated gradient mask shows the outline of the cell quite nicely, but
% there are still holes in the interior of the cell. To fill these holes we
% use the imfill function.

BWdfill = imfill(BWsdil, 'holes');
% figure, imshow(BWdfill);
% title('binary image with filled holes');

%% Step 5: Remove Connected Objects on Border
% The cell of interest has been successfully segmented, but it is not the
% only object that has been found. Any objects that are connected to the
% border of the image can be removed using the imclearborder function. The
% connectivity in the imclearborder function was set to 4 to remove
% diagonal connections.

BWnobord = imclearborder(BWdfill, 4);
% figure, imshow(BWnobord), title('cleared border image');

%% Step 6: Smoothen the Object
% Finally, in order to make the segmented object look natural, we smoothen
% the object by eroding the image twice with a diamond structuring element.
% We create the diamond structuring element using the |strel| function.

seD = strel('diamond',1);
BWfinal = imerode(BWnobord,seD);
BWfinal = imerode(BWfinal,seD);
figure, imshow(BWfinal), title('segmented image');
