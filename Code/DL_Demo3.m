close all;
clear all;
% Read in image
I = imread('chess9.jpg');

% Scale down 50%
I = imresize(I, 0.5);

% Convert to grayscale
Ig = rgb2gray(I);

% Get a gaussian kernel for blurring
K = fspecial('gaussian');

% Blur the image
% Note that multiple passes with a fixed kernel
% are the same as a single pass with a larger kernel
Igf = imfilter(Ig, K);
Igf = imfilter(Igf, K);
Igf = imfilter(Igf, K);

% Detect edges
E = edge(Ig, 'sobel');
se = strel('line',6,90);
se1 = strel('line',6,0);
% afterOpening = imopen(E,se);
afterOpening = imdilate(E,se);
afterOpening = imdilate(afterOpening,se1);

% Perform Hough Line transform
[H, T, R] = hough(afterOpening);

% Get top N line candidates from hough accumulator
N = 20;
P = houghpeaks(H, N);

% Get hough line parameters
lines = houghlines(Igf, T, R, P);

% Show lines overlaid on original image, and hough space
figure;

% Show original image
subplot(2, 1, 1);
imshow(I);
title('Detected Lines');
hold on;

coeff =[];
% Overlay detected lines - this code copied from 'doc houghlines'
for k = 1:length(lines)
   xy = [lines(k).point1; lines(k).point2];
   lines_int = plot(xy(:,1), xy(:,2), 'LineWidth', 2, 'Color', 'blue');
   
end

figure;
subplot(2, 1, 1);
imshow(I);
title('Intersecting Points');
hold on;
pointInt = [];
for k = 1:length(lines)
     xy = [lines(k).point1(1) lines(k).point2(1);lines(k).point1(2) lines(k).point2(2)];
    for l = 1:length(lines)
        if(k~=l)
            xy1 = [lines(l).point1(1) lines(l).point2(1);lines(l).point1(2) lines(l).point2(2)];
            inter = InterX(xy,xy1);
            x = inter(1,:);
            y = inter(2,:);
            if (isempty(x) && isempty(y))
            else
                pointInt = [pointInt; [x y]];
            end
            
        end
    end
end

pointInt = unique(pointInt, 'rows');
plot(pointInt(:,1),pointInt(:,2),'ro');

k = boundary(pointInt(:,1),pointInt(:,2),1);
plot(pointInt(k),pointInt(k));

addpath('matching');

corners = findCorners(Ig,0.05,2);
chessboards = chessboardsFromCorners(corners);

figure; imshow(uint8(Ig)); hold on;
plotChessboards(chessboards,corners);

figure;
board = chessboards{1,1};
k=1;

bboxfinal = [];
for i=1:size(board,1)-1
    for j=1:size(board,2)-1
        points = [];
        a = board(i,j);
        b = board(i,j+1);
        c = board(i+1,j);
        d = board(i+1,j+1);
        
        points = [points; corners.p(a,:)];
        points = [points; corners.p(b,:)];
        points = [points; corners.p(c,:)];
        points = [points; corners.p(d,:)];
        
        bb = boundingBox(points);
        xmin = bb(1,1);
        ymin = bb(1,3);
        width = bb(1,2)-bb(1,1);
        height = bb(1,4)-bb(1,3);
        bboxfinal = [bboxfinal;[xmin, ymin, width, height]];
        
        subplot(5,11,k);
        I2 = imcrop(I,[xmin, ymin, width, height]);
        imshow(I2);
        filename = (['cropped_',num2str(k),'.jpg']);
        imwrite( I2, filename );
        k=k+1;
    end
end

%extend bounding boxes
edgePoint = board(1,1);
edgePoint = corners.p(edgePoint,:);
bb1 = [edgePoint(1,1)-width,edgePoint(1,2),width,height];
bb2 = [edgePoint(1,1)-width,edgePoint(1,2)-(height*1.1),width,height*1.1];
bboxfinal = [bboxfinal;bb1];
bboxfinal = [bboxfinal;bb2];
I2 = imcrop(I,bb1);
I3 = imcrop(I,bb2);
filename = (['cropped_',num2str(k),'.jpg']);
imwrite( I2, filename );
subplot(5,11,k);
imshow(I2);
k=k+1;
filename = (['cropped_',num2str(k),'.jpg']);
imwrite( I3, filename );
subplot(5,11,k);
imshow(I3);
k=k+1;

bb2(1,1)=bb2(1,1)-20;

for i=1:7
bb2(1,1) = (bb2(1,1)+width);
bb2(1,2) = bb2(1,2);
bboxfinal = [bboxfinal;bb2];
I2 = imcrop(I,bb2);
filename = (['cropped_',num2str(k),'.jpg']);
subplot(5,11,k);
imshow(I2);

imwrite( I2, filename );
k=k+1;
end

edgePoint1 = board(7,1);
edgePoint1 = corners.p(edgePoint1,:);
bb1 = [edgePoint1(1,1)-width,edgePoint1(1,2),width,height];
bb2 = [edgePoint1(1,1)-width,edgePoint1(1,2)-(height*1.1),width,height*1.1];
I2 = imcrop(I,bb1);
I3 = imcrop(I,bb2);

subplot(5,11,k);
imshow(I2);

filename = (['cropped_',num2str(k),'.jpg']);
imwrite( I2, filename );
bboxfinal = [bboxfinal;bb1];
bboxfinal = [bboxfinal;bb2];
k=k+1;
filename = (['cropped_',num2str(k),'.jpg']);
imwrite( I3, filename );
subplot(5,11,k);
imshow(I3);
k=k+1;

for i=1:7
bb1(1,1) = (bb1(1,1)+width);
bb1(1,2) = bb1(1,2);
bboxfinal = [bboxfinal;bb1];
I2 = imcrop(I,bb1);
filename = (['cropped_',num2str(k),'.jpg']);
imwrite( I2, filename );
subplot(5,11,k);
imshow(I2);
k=k+1;
end



%%%%Recognition Part Begins


rootFolder = fullfile('chess');
categories = {'black_bishop', 'black_king', 'black_queen', 'black_knight', 'black_castle', 'black_pawn', 'white_bishop', 'white_king', 'white_queen', 'white_knight', 'white_castle', 'white_pawn','empty'};

imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');

tbl = countEachLabel(imds)

minSetCount = min(tbl{:,2}); % determine the smallest amount of images in a category

% Use splitEachLabel method to trim the set.
imds = splitEachLabel(imds, minSetCount, 'randomize');

% Notice that each set now has exactly the same number of images.
countEachLabel(imds)


% Store CNN model in a temporary folder
cnnMatFile = fullfile('imagenet-caffe-alex.mat');

convnet = helperImportMatConvNet(cnnMatFile)

%%
% |convnet.Layers| defines the architecture of the CNN. 

% View the CNN architecture
convnet.Layers

% Inspect the first layer
convnet.Layers(1)

% Inspect the last layer
convnet.Layers(end)

% Number of class names for ImageNet classification task
numel(convnet.Layers(end).ClassNames)

imds.ReadFcn = @(filename)readAndPreprocessImage(filename);



[trainingSet, testSet] = splitEachLabel(imds, 0.3, 'randomize');

w1 = convnet.Layers(2).Weights;

% Scale and resize the weights for visualization
w1 = mat2gray(w1);
w1 = imresize(w1,5); 

% Display a montage of network weights. There are 96 individual sets of
% weights in the first layer.
figure
montage(w1)
title('First convolutional layer weights')

featureLayer = 'fc7';
trainingFeatures = activations(convnet, trainingSet, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');

trainingLabels = trainingSet.Labels;

classifier = fitcecoc(trainingFeatures, trainingLabels, ...
    'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');


% Extract test features using the CNN
testFeatures = activations(convnet, testSet, featureLayer, 'MiniBatchSize',32);

% Pass CNN image features to trained classifier
predictedLabels = predict(classifier, testFeatures);

% Get the known labels
testLabels = testSet.Labels;

% Tabulate the results using a confusion matrix.
confMat = confusionmat(testLabels, predictedLabels);

% Convert confusion matrix into percentage form
confMat = bsxfun(@rdivide,confMat,sum(confMat,2))
%%

% Display the mean accuracy
mean(diag(confMat))

I_final = I;

for i=1:53
    
    filename = (['cropped_',num2str(i),'.jpg']);
    img = readAndPreprocessImage(filename);
    disp(i);
    % Extract image features using the CNN]
    imageFeatures = activations(convnet, img, featureLayer);
    %%

    % Make a prediction using the classifier
    label = predict(classifier, imageFeatures)
    
     I_final = insertObjectAnnotation(I_final, 'rectangle', bboxfinal(i+1,:), char(label));
    
end

figure;
imshow(I_final);

title('Image with Detected Chess Pieces');


