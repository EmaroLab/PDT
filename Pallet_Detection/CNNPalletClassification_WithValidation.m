% This script presents:
%      1. The main steps for training the dataset using the CNN classifier and defines the convolutional neural network architecture
%         in order to classify Pallet among other negatives tracks with high confidence scores. 
%      2. It also applies the k-fold cross-validation to evaluate the
%         performance of the proposed system.

clc; close all; clear variables;

% Set the parameters
KFold = 5;         ImSize = 250;           Ch = 3;             MaxEpochs = 10;
% In order to increase the pallet dataset for training by considering the 340 images with only pallet region. 
AllRGB = true;

if AllRGB
    % For 340 pallet images
    load('AllData/PalletImages/RGBImages/PalletRGBImages/Pallet250size/CNNDataForPose.mat');
    Data1 = RGBImages';
    load('AllData/PalletImages/RGBImages/PalletRGBImages/Pallet250size/DataForSecondCNN.mat');
    Data2 = RGBImages';
    % The whole data of class 1 (Pallet) which is 450 images
    PalletData = [Data1; Data2(1:110,:)];
    % The whole data of class 2 (No Pallet) which is 500 images
    NoPalletData = Data2(111:end,:);
    % The dataset for both classes
    DataSet = [PalletData; NoPalletData];
else
    % Load RGB Images for Second CNN) == 110 for pallet, and the rest for non pallet
    load('AllData/PalletImages/RGBImages/PalletRGBImages/Pallet250size/DataForSecondCNN.mat')
    DataSet = RGBImages';
end

% Extract both classes in order to find the training and the testing sets
if AllRGB
    PalletClass = DataSet(1:length(PalletData));
    NoPalletClass = DataSet(length(PalletData)+1:end);
else
    PalletClass = DataSet(1:110);
    NoPalletClass = DataSet(111:end);
end
% Set the corresponding Class labels (Targets)
Labels = [ones(1,length(PalletClass)), -1*ones(1,length(NoPalletClass))]';

% Generate cross-validation indices
Indices = crossvalind('Kfold',Labels,KFold);
accuracy = zeros(KFold,1);
for i = 1:KFold
    
    TestingIdx = find(Indices == i);                    TrainingIdx = find(Indices~=1);
    TestingData = DataSet(TestingIdx,:);                TrainingData = DataSet(TrainingIdx,:);
    TestingLabels = Labels(TestingIdx,:);               TrainingLabels = Labels(TrainingIdx,:);
    TestingLabels = categorical(TestingLabels);         TrainingLabels = categorical(TrainingLabels);
    
    % convert 2D cell to 4D array.
    % The 1st three dimensions must be the height, width, and channels, the last one is index of individual images.
    TrainData4D = reshape(cat(3,TrainingData{:}),ImSize,ImSize,Ch,length(TrainingData));
    TrainData4D = im2double(TrainData4D);

    TestData4D = reshape(cat(3,TestingData{:}),ImSize,ImSize,Ch,length(TestingData));
    TestData4D = im2double(TestData4D);

    %% Create a Convolutional Neural Network (CNN)

    % Define the convolutional neural network architecture
    layers = [imageInputLayer([ImSize ImSize Ch])
            convolution2dLayer(20,25) % 25= # of filters
            reluLayer
            %convolution2dLayer(5,30)
            %reluLayer
            maxPooling2dLayer(5,'Stride',2)
            fullyConnectedLayer(2)
            softmaxLayer
            classificationLayer()]; 
     
    options = trainingOptions('sgdm','MaxEpochs',MaxEpochs, ...
        'InitialLearnRate',0.03, ...
        'MiniBatchSize',50);

    [ConvNet, traininfo] = trainNetwork(TrainData4D,TrainingLabels,layers,options);

    % Run the trained network on the test set 
    YTest = classify(ConvNet,TestData4D);
    TTest = TestingLabels;

    % Calculate the accuracy.
    accuracy(i,:) = sum(YTest == TTest)/numel(TTest);
end
% Calculate the average accuracy of the kfold validation step. 
Av_Accuracy = mean(accuracy);

%% The next step after Kfold validation phase is to consider all data set for training.

% convert 2D cell to 4D array.
% The 1st three dimensions must be the height, width, and channels, the last one is index of individual images.
DataSet4D = reshape(cat(3,DataSet{:}),ImSize,ImSize,Ch,length(DataSet));
DataSet4D = im2double(DataSet4D);
Labels = categorical(Labels);

% Create a Convolutional Neural Network (CNN)

% Define the convolutional neural network architecture
layers = [imageInputLayer([ImSize ImSize Ch])
        convolution2dLayer(20,25)
        reluLayer
        %convolution2dLayer(5,30)
        %reluLayer
        maxPooling2dLayer(5,'Stride',2)
        fullyConnectedLayer(2)
        softmaxLayer
        classificationLayer()]; 

options = trainingOptions('sgdm','MaxEpochs',MaxEpochs, ...
    'InitialLearnRate',0.03, ...
    'MiniBatchSize',50);

% The network will be used for testing the new data (images).
ConvNet = trainNetwork(DataSet4D,Labels,layers,options);



