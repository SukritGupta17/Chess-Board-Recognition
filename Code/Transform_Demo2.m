close all;
clear all;
% Read in image
I = imread('chess2.jpg');

% Scale down 50%
I = imresize(I, 0.5);

f1 = figure;
imshow(I);
title('Original Input Chessboard Image');
% Convert to grayscale
Ig = rgb2gray(I);

% Get a gaussian kernel for blurring
K = fspecial('gaussian');

% Blur the image
% Note that multiple passes with a fixed kernel
% are the same as a single pass with a larger kernel
f2 = figure;
subplot(1, 1, 1);
Igf = imfilter(Ig, K);
Igf = imfilter(Igf, K);
Igf = imfilter(Igf, K);
imshow(Igf);
title('Gaussian Filtered Image');



f3 = figure;
subplot(1, 1, 1);
% Detect edges
E = edge(Ig, 'canny');
imshow(E);
title('Edge Detected Image');

% Perform Hough Line transform
[H, T, R] = hough(E);

% Get top N line candidates from hough accumulator
N = 25;
P = houghpeaks(H, N);

% Get hough line parameters
lines = houghlines(Igf, T, R, P);


coeff =[];
figure;
% Overlay detected lines - this code copied from 'doc houghlines'
for k = 1:length(lines)
   xy = [lines(k).point1; lines(k).point2];
   plot(xy(:,1), xy(:,2), 'LineWidth', 2, 'Color', 'blue');
   
   % Plot beginnings and ends of lines
   plot(xy(1,1), xy(1,2), 'x', 'LineWidth', 2, 'Color', 'yellow');
   plot(xy(2,1), xy(2,2), 'x', 'LineWidth', 2, 'Color', 'red');
   
end

f4 = figure;
subplot(1, 1, 1);
imshow(I);
title('Detected Lines - Hough Transform');
hold on;

pointInt = [];
for k = 1:length(lines)
     xy = [lines(k).point1; lines(k).point2];
    for l = 1:length(lines)
        if(k~=l)
            xy1 = [lines(l).point1; lines(l).point2];
            [x, y] = lineintersect([xy(1,1) xy(1,2) xy(2,1) xy(2,2)],[xy1(1,1) xy1(1,2) xy1(2,1) xy1(2,2)]);
            if (isnan(x) && isnan(y))
            else
                pointInt = [pointInt; [x y]];
            end
            
        end
    end
end
plot(pointInt(:,1),pointInt(:,2),'go');

k = boundary(pointInt(:,1),pointInt(:,2));
hold on;
f5 = figure;
subplot(2, 1, 1);
imshow(I);
title('Detected Boundary');
hold on;
f6 = figure;
x = pointInt(:,1);
y = pointInt(:,2);
A = plot(x(k),y(k), 'LineWidth', 3, 'Color', 'green');
axis off;
saveas(A,'myimage.jpg');
bboximg = imread('myimage.jpg');
bboximg = flipdim(bboximg,1); 

temp = rgb2gray(bboximg);
temp1 = imbinarize(temp);

f7 = figure;
subplot(1,1,1);
cornerPoints = corner(temp1,4);
imshow(temp1);
figure;
plot(cornerPoints(:,1), cornerPoints(:,2), 'r*');
fp = matfile('cornet.mat');
cornerPoints = fp.movingPoints;
title('Corner Points');

% Show Hough Space

% imshow(imadjust(mat2gray(H)), 'XData', T, 'YData', R, 'InitialMagnification', 'fit');
% title('Hough Line Transform');
% xlabel('\theta');
% ylabel('\rho');
% axis on;
% axis normal;
% grid on;
% hold on;

% Display as colormap
% colormap('jet');

fp = matfile('fixedPoints.mat');
fixedPoints = fp.fixedPoints;

% Use the selected points to create a recover the projective transform
tform = cp2tform(cornerPoints, fixedPoints, 'projective');

f8 = figure;
subplot(1,1,1);
% Transform the grayscale image
Igft = imtransform(I, tform, 'XYScale', 0.2);
imshow(Igft);
title('Transformed Image');

