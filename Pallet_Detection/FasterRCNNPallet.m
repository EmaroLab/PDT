% This script shows:
%        1. How to train and test the Fast R-CNN detector using the ready data 
%                which we collected manually using "Image Labeler App" 
%        2. How to extract ROIs from the given image to train the CNN classifier. 

clc; close all; clear variables;

%% Load Pallet images (Gray and RGB Images)

S = rng(1); % For shuffling data
 
PalletDatasetWithRotation  = true;  % A logical parameter to consider the artificial data 
                                    % (by rotating the original image by +90 & -90 degrees) and the original
                                    % dataset for training the Faster R-CNN
                                    
InOrder = true;                     % A logical parameter to consider the whole dataset in order 
                                    % without overlaping between the ROIs.
                                    
GrayIm = false;                     % A logical parameter to use the Gray images dataset for training.

Calculate_Precision = 0;         % A logical parameter to calculate the Average Precision (AP).
                                    % Note that: We evalute the performance of our system after performing calssification using CNN.
                                    % In other words, AP will not be used for evaluating the performance

doTrainingAndEval = 0;          % A logical parameter for retraining the Faster R-CNN.
DataForNextNetwork = 0;         % A logical parameter to prepare data and extract ROIs from the given image to train the CNN classifier.

if PalletDatasetWithRotation
    % In the case of extracting the non pallet part of the images and adding the rotated images to the dataset
    if GrayIm
        load('AllData/PalletImages/PalletDataWithRotation_GrayImage.mat')
        %load('AllData/PalletImages/PalletDataWithRotation_GrayImage_NoOverlaps.mat') 
        
        % In order to set the full path for images
        PalletDataset.PalletFileImages = erase(PalletDataset.PalletFileImages, '/home/ihab/MyData/Master_EMARO/MasterThesis/Matlab Codes/SVM/Pallet_Detection/');
        % Be sure that you are in "Pallet_Detection" folder.
        currentfolder = cd;
        % Add fullpath to the local pallet data folder.
        PalletDataset.PalletFileImages = fullfile(currentfolder, PalletDataset.PalletFileImages);
    else
        if InOrder
            load('AllData/PalletImages/RGBImages/PalletRGBImages/Pallet250size/WholePalletDataWithRotation_RGBImage_NoOverlaps_InOrder.mat')
        else
            %load('AllData/PalletImages/RGBImages/PalletRGBImages/Pallet250size/PalletDataWithRotation_RGBImage.mat')
            load('AllData/PalletImages/RGBImages/PalletRGBImages/Pallet250size/PalletDataWithRotation_RGBImage_NoOverlaps.mat')
        end
        % In order to set the full path for images
        PalletDataset.PalletFileImages = erase(PalletDataset.PalletFileImages, '/home/ihab/MyData/Master_EMARO/MasterThesis/Matlab Codes/SVM/Pallet_Detection/');
        % Be sure that you are in "Pallet_Detection" folder.
        currentfolder = cd;
        % Add fullpath to the local pallet data folder.
        PalletDataset.PalletFileImages = fullfile(currentfolder,PalletDataset.PalletFileImages);
    end
else
    % Load RGB/Gray Image information table without considering the artifical data.
    if GrayIm
        load('AllData/PalletImages/PalletGrayImages/GrayPalletDataset.mat');
        % In order to set the full path for images
        PalletDataset.PalletFileImages = erase(PalletDataset.PalletFileImages, '/home/ihab/MyData/Master_EMARO/MasterThesis/Matlab Codes/SVM/Pallet_Detection/');
        % Be sure that you are in "Pallet_Detection" folder.
        currentfolder = cd;
        % Add fullpath to the local pallet data folder.
        PalletDataset.PalletFileImages = fullfile(currentfolder, PalletDataset.PalletFileImages);
    else
        load('AllData/PalletImages/RGBImages/PalletRGBImages/Pallet250size/RGBPalletDataset.mat')
        % In order to fix the full path for images
        PalletDataset.PalletFileImages = erase(PalletDataset.PalletFileImages, '/home/ihab/MyData/Master_EMARO/MasterThesis/Matlab Codes/SVM/Pallet_Detection/');
        % Be sure that you are in "Pallet_Detection" folder.
        currentfolder = cd;
        % Add fullpath to the local pallet data folder.
        PalletDataset.PalletFileImages = fullfile(currentfolder, PalletDataset.PalletFileImages);
    end
end

%%  Split the dataset into a training (70%) and testing set.

% PalletIdx = randperm(height(PalletDataset));
Original = 340; Rotated = 679;
idx = floor(0.7 * height(PalletDataset));
idx1 = floor(0.7 * Original); idx2 = floor(0.7 * Rotated); 

if PalletDatasetWithRotation
    if InOrder
        TrainingData1 = PalletDataset(1:idx1,:);
        TrainingData2 = PalletDataset(Original+1:Original+idx2,:);
        TrainingData = [TrainingData1; TrainingData2];
        TrainingData = TrainingData(:,1:2);

        TestingData1 = PalletDataset(idx1+1:Original,:);
        TestingData2 = PalletDataset(Original+idx2+1:Original+Rotated,:);
        TestingData  = [TestingData1; TestingData2];
        TestingData = TestingData(:,1:2);
    else
        TrainingData1 = PalletDataset(1:idx1,:);
        TrainingData2 = PalletDataset(Original+1:Original+idx2,:);
        TrainingData = [TrainingData1; TrainingData2];
        TrainingData = TrainingData(:,1:2);

        TestingData1 = PalletDataset(idx1+1:Original,:);
        TestingData2 = PalletDataset(Original+idx2+1:Original+Rotated,:);
        TestingData  = [TestingData1; TestingData2];
        TestingData = TestingData(:,1:2);
    end
else
    TrainingData = PalletDataset(1:idx,:);
    TrainingData = TrainingData(:,1:2);

    TestingData = PalletDataset(idx+1:end,:);
    TestingData = TestingData(:,1:2);
end

%% Read one of the given images.

% ImageIdx = randi([1 height(PalletDataset)],1,1);
 ImageIdx = 2;
if PalletDatasetWithRotation
    I = imread(PalletDataset.PalletFileImages{ImageIdx});

    % Insert the ROI labels.
    I = insertShape(I, 'Rectangle', PalletDataset.PalletBox{ImageIdx},'Color','green','Opacity',0.7);
    % I = insertShape(I,'Rectangle',PalletDataset.NonPalletBox{ImageIdx},'Color','red','Opacity',0.7); 
    
    % Resize and display image.
    I = imresize(I, 1.5);
    figure
    imshow(I)
    title('One of the input images to the Faster R-CNN with the ROIs')
else
    I = imread(PalletDataset.PalletFileImages{ImageIdx});

    % Insert the ROI labels.
    I = insertShape(I, 'Rectangle', PalletDataset.PalletBox{ImageIdx},'Color','green','Opacity',0.7);
    % I = insertShape(I,'Line',PalletDataset.PalletWidth{ImageIdx},'LineWidth',2,'Color','red'); % For width 

    % Resize and display image.
    I = imresize(I, 1.5);
    figure
    imshow(I)
    title('One of the input images to the Faster R-CNN with the ROIs')
end
    
%% Create a Convolutional Neural Network (CNN)

ImageSize = 32;
% Create image input layer.
if GrayIm
    inputLayer = imageInputLayer([ImageSize ImageSize 1]); % In case of Gray Images
else
    inputLayer = imageInputLayer([ImageSize ImageSize 3]); % In case of GRB Images
end

% Define the convolutional layer parameters.
% filterSize1 = [5 5];
filterSize = [3 3];
numFilters = 40;

% Create the middle layers.
middleLayers = [
    convolution2dLayer(filterSize, numFilters, 'Padding', 1)
    reluLayer()
    convolution2dLayer(filterSize, numFilters, 'Padding', 1)
    reluLayer()
    maxPooling2dLayer(3, 'Stride',1)
    ];

% Create the final layers.
finalLayers = [
    fullyConnectedLayer(64)
    reluLayer()
    fullyConnectedLayer(width(TrainingData))
    softmaxLayer()
    classificationLayer()
];

layers = [
    inputLayer
    middleLayers
    finalLayers
    ];

% Options for step 1.
optionsStage1 = trainingOptions('sgdm', ...
    'MaxEpochs', 10, ...
    'InitialLearnRate', 1e-5, ...
    'CheckpointPath', tempdir);

% Options for step 2.
optionsStage2 = trainingOptions('sgdm', ...
    'MaxEpochs', 10, ...
    'InitialLearnRate', 1e-5, ...
    'CheckpointPath', tempdir);

% Options for step 3.
optionsStage3 = trainingOptions('sgdm', ...
    'MaxEpochs', 15, ...
    'InitialLearnRate', 1e-6, ...
    'CheckpointPath', tempdir);

% Options for step 4.
optionsStage4 = trainingOptions('sgdm', ...
    'MaxEpochs', 15, ...
    'InitialLearnRate', 1e-6, ...
    'CheckpointPath', tempdir);

options = [
    optionsStage1
    optionsStage2
    optionsStage3
    optionsStage4
    ];

if doTrainingAndEval
    % Train Faster R-CNN detector.
    detector = trainFasterRCNNObjectDetector(PalletDataset(:,1:2), layers, options, ...
            'NegativeOverlapRange', [0 0.3]); %'BoxPyramidScale', 1.2); 'NegativeOverlapRange', [0.1 0.3], ... 
else
    % Load the pre-trained detector for evaluate the performancen of the network.
    if PalletDatasetWithRotation
        if GrayIm
            data = load('PreTrained_CNNs/TrainedFasterR-CNN/FastCNNPalletWhole340.mat');
        else
            % data = load('TrainedFastR-CNN/FastCNNPallet340RGBImages14.mat');
            data = load('PreTrained_CNNs/TrainedFasterR-CNN/FastCNNPallet340RGBImages26.mat');
        end
    else
        if GrayIm
            data = load('PreTrained_CNNs/TrainedFasterR-CNN/FastCNNPalletWhole340.mat');
            % data = load('TrainedFastR-CNN/FastCNNPallet250size.mat'); % case considering the front part of pallet only
        else
            % data = load('TrainedFastR-CNN/FastCNNPalletWhole340.mat');
            data = load('PreTrained_CNNs/TrainedFasterR-CNN/FastCNNPallet340RGBImages26.mat');
        end
    end
   detector = data.detector;
end

%% Testing the Faster R-CNN using the testing dataset 

if doTrainingAndEval
    % Run detector on each image in the testing dataset and save the results.
    resultsStruct = struct([]);
    for i = 1:height(TestingData)
        % Read the image.
        I = imread(TestingData.PalletFileImages{i});
        % Run the detector.
        [bboxes, scores, labels] = detect(detector, I);
        % Select the strongest regions with 0.3 overlap Threshold 
        [selectedBbox,selectedScore,selectedIdx] = selectStrongestBbox(bboxes,scores,'OverlapThreshold', 0.3);
        % Collect the results.
        resultsStruct(i).Boxes = bboxes;
        resultsStruct(i).Scores = scores;
        resultsStruct(i).Labels = labels;
        
        resultsStruct(i).SelectedBoxes = selectedBbox;
        resultsStruct(i).SelectedScores = selectedScore;
        resultsStruct(i).SelectedLabels = selectedIdx;
        
        % Select the ROI with the maximum confidence score.
        [score, MaxIdx] = max(scores); 
        resultsStruct(i).MaxScores = score;
        resultsStruct(i).MaxIdx = MaxIdx;
        pause(0.2)
    end
    % Convert the results into a table.
    results = struct2table(resultsStruct);
else
    % Load results from disk.
    results = data.results;
end

%% Read one of the test images.

% TestIdx = randi([1 height(TestingData)],1,1);
TestIdx = 200;
TestImage = imread(TestingData.PalletFileImages{TestIdx});
figure
imshow(TestImage)
title('One of the test Images')

% Run the detector.
[bboxes, scores,label] = detect(detector,TestImage);

% Annotate detections in the image.
I = insertObjectAnnotation(TestImage, 'rectangle', bboxes, scores);
figure
I = imresize(I, 1.5);
imshow(I)
title('The test Image with the detected ROIs')

% Select the strongest detected objects
[selectedBbox,selectedScore,selectedIdx] = selectStrongestBbox(bboxes,scores,'OverlapThreshold', 0.2);

% Annotate detections in the image.
I = insertObjectAnnotation(TestImage, 'rectangle', selectedBbox, selectedScore);
figure
I = imresize(I, 1.5);
imshow(I)
title('Detected objects and detection scores after suppression')

% Display the max. score detected object
[score, Idx] = max(scores);

bbox = bboxes(Idx, :);
annotation = sprintf('%s: (Confidence = %f)', label(Idx), score);

outputImage = insertObjectAnnotation(TestImage, 'rectangle', bbox, annotation);

% figure
% imshow(outputImage)
% title('The max. score detected object')
%{
croppedImage = imcrop(TestImage, bbox);
% figure;
% imshow(croppedImage);

% Image with only the ROIs 
whiteImage = 255 * ones(250, 250, 3, 'uint8');
whiteImage(bbox(2):bbox(2)+bbox(4),bbox(1):bbox(1)+bbox(3),:) = croppedImage;
figure;
imshow(whiteImage);
title('The test Image with only ROI')
%}

%% Evaluate the object detector using Average Precision metric.

% Extract expected bounding box locations from test data.
expectedResults = TestingData(:,2);
if Calculate_Precision
    
    % Evaluate the object detector using Average Precision metric.
    [ap, recall, precision] = evaluateDetectionPrecision(results(:,4:5), expectedResults, 0);
    % Plot precision/recall curve
    figure
    plot(recall, precision)
    xlabel('Recall')
    ylabel('Precision')
    grid on
    title(sprintf('Average Precision = %.1f', ap))
end

%% In order to prepare data and extract ROI from the given image to train the next network (CNN)
if DataForNextNetwork
    load('PreTrained_CNNs/TrainedFasterR-CNN/FastCNNPallet340RGBImages26.mat');
    for i = 1:height(TestingData)
        % Use the Strongest ROIs.
        RegionBoxes = results.SelectedBoxes{i};
        RegionsScore = results.SelectedScores{i};
        
        TestImage = imread(TestingData.PalletFileImages{i});
        Loop = 'Test Image No. %d\n'; fprintf(Loop,i);
        % figure; imshow(TestImage)
        TestImageWithRegions = imread(sprintf('AllData/PalletImages/RGBImages/PalletRGBImages/Pallet250size/StrongestRegions/%d.png', i));
        % figure; imshow(TestImageWithRegions)
        StrongestRegionsImage = insertObjectAnnotation(TestImage, 'rectangle', RegionBoxes, RegionsScore);
        % figure; I = imresize(StrongestRegionsImage, 1.5); imshow(I)
        
        if ~isempty(RegionBoxes)
           for B = 1:size(RegionBoxes,1)
                bbox = RegionBoxes(B,:);
                croppedImage = imcrop(TestImage, bbox);
                % imshow(croppedImage)
                % Image with only interested points 
                whiteImage = 255 * ones(250, 250, 3, 'uint8');
                % Check the Image borders
                Width = bbox(2)+bbox(4); Height = bbox(1)+bbox(3);
                if Width>250
                    Width = 250;
                end
                if Height>250
                    Height =250;
                end
                whiteImage(bbox(2):Width,bbox(1):Height,:) = croppedImage;
                imwrite(whiteImage,sprintf('AllData/PalletImages/RGBImages/PalletRGBImages/Pallet250size/ExtractedRegions/%d_%d.png', i,B))
                pause(0.2)
                clf, close all;
            end 
        end
    end
end
