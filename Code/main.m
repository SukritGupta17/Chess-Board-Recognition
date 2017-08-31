I = imread('chess2.jpg');
I = rgb2gray(I);
level = graythresh(I);
BW = imbinarize(I,level);
% [imagePoints,boardSize] = detectCheckerboardPoints(I);
BW = edge(BW,'Canny',0);
imshow(BW);

% Perform Hough Line transform
[H, T, R] = hough(BW);

% Get top N line candidates from hough accumulator
N = 10;
P = houghpeaks(H, N);

% Get hough line parameters
lines = houghlines(BW, T, R, P);

% Show lines overlaid on original image, and hough space
figure;

% Show original image
subplot(2, 1, 1);
imshow(I);
title('chess\_circles.jpg');
hold on;

% Overlay detected lines - this code copied from 'doc houghlines'
for k = 1:length(lines)
   xy = [lines(k).point1; lines(k).point2];
   plot(xy(:,1), xy(:,2), 'LineWidth', 2, 'Color', 'blue');

   % Plot beginnings and ends of lines
   plot(xy(1,1), xy(1,2), 'x', 'LineWidth', 2, 'Color', 'yellow');
   plot(xy(2,1), xy(2,2), 'x', 'LineWidth', 2, 'Color', 'red');
end
% subplot(2, 2, 1);
% imshow(I);
% hold on;
% plot(imagePoints(:,1,1),imagePoints(:,2,1),'ro');