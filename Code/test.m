I = imread('chess8.jpg');
I = rgb2gray(I);
thresh = multithresh(I,6);


seg_I = imquantize(I,thresh);

RGB = label2rgb(seg_I); 	 
figure;
imshow(RGB)