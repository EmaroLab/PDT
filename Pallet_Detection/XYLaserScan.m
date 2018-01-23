%% This script shows how to convert the acquired data from Polar to Cartesian coordinates.

clc; close all; clear;

%% Set Laser scanner parameters

NumOfSamples = 761;
FOV = 190;
FOV = deg2rad(FOV);
res = FOV/NumOfSamples;
StartAngle = -FOV/2;
EndAngle = FOV/2;
% The angle vector which corresponding to the measured distance from an obstacle to the laser
AngleVector = zeros(1,NumOfSamples);

for s = 1: NumOfSamples
    AngleVector(1,s) = StartAngle +res*(s-1);
end

%% load the Data Files.

% Class1: Pallet class 
ScanPalletFiles = 340; ScanPoints = 761;
 
PalletClass = cell(1, ScanPalletFiles);
Dataset1 = zeros(ScanPoints, ScanPalletFiles);

% Call all files one by one and save it in cell
for i = 1:ScanPalletFiles
  myfilename = sprintf('AllData/Class1/Scan%d.txt', i);
  PalletClass{i} = importdata(myfilename);
end

for i = 1 : ScanPalletFiles
    % Call all files one by one in the cell
    Class1_Dataset = PalletClass{i};
    % Extract the dataset coloumn
    Class1_Dataset = Class1_Dataset(:,2);
    % Note that: Each .txt file contains 4 frames with a total number of
    % 761 scans points. For that reason we have to take the average over
    % the rows.
    Class1_Dataset = reshape(Class1_Dataset,[761,4]);
    % Take the mean over rows
    av_dataset = mean(Class1_Dataset,2);
    % Save all Training  dataset
    Dataset1(:,i) = av_dataset;
end

%% Class2: NoN-Pallet class 

ScanNonPalletFiles = 225;
NonPalletClass = cell(1, ScanNonPalletFiles);
Dataset2 = zeros(ScanPoints, ScanNonPalletFiles);

for j = 1:ScanNonPalletFiles
  myfilename = sprintf('AllData/Class2/Scan%d.txt', j);
  NonPalletClass{j} = importdata(myfilename);
end

for i = 1 : ScanNonPalletFiles
    % Call all files one by one in the cell
    Class2_Dataset = NonPalletClass{i};
    % Extract the data set coloumn
    Class2_Dataset = Class2_Dataset(:,2);
    Class2_Dataset = reshape(Class2_Dataset,[761,4]);
    % Take the mean over rows
    av_dataset2 = mean(Class2_Dataset,2);
    % Save all Training  dataset 
    Dataset2(:,i) = av_dataset2;
end

%% DataSets of both classes

Dataset1 = Dataset1';
Dataset2 = Dataset2';
% combine both datasets
Dataset = [Dataset1;Dataset2];

%% Find the corresponding XY points

x =zeros(size(Dataset,1),size(Dataset,2));
y =zeros(size(Dataset,1),size(Dataset,2));

for row = 1:size(Dataset,1)
    x(row,:) = Dataset(row,:).*cos(AngleVector);
    y(row,:) = Dataset(row,:).*sin(AngleVector);
end

% plot one of the acquired data (e.g. scene No 10)  
figure(1)
c = linspace(1,13,length(Dataset));
scatter(AngleVector, Dataset(10,:) , c, 'filled')

title('Scan Points in Polar Coordinates')
xlabel('\theta')
ylabel('Distance d')

figure(2)
c = linspace(1,13,length(Dataset));
scatter(y(10,:),x(10,:), c, 'filled')
title('Scan Points in Cart Coordinates')
xlabel('Y')
ylabel('X')

