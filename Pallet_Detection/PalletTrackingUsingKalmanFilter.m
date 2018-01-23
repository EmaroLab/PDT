function PalletTrackingUsingKalmanFilter()
% This function shows how to detect pallets using 2D laser scanner and keep tracking all objects (pallets) in the laser FOV. 
% It is composed by:
%   1. Fast R-CNN for detecting all regions of interest in the environment 
%   2. CNN for extracting and checking whether the pallet is one of the detected regions or not.
%   3. using the Kalman filter for tracking the objects.

% Set the global parameters.
option.gatingThresh         = 0.9;              % A threshold to reject a candidate match between a detection and a track.
option.gatingCost           = 50;               % A large value for the assignment cost matrix that enforces the rejection of a candidate match.
option.costOfNonAssignment  = 100;              % A tuning parameter to control the likelihood of creation of a new track.
option.timeWindowSize       = 6;                % A tuning parameter to specify the number of frames required to stabilize the confidence score of a track.
option.confidenceThresh     = 0.6;              % A threshold to determine if a track is true positive or false alarm.
option.MinconfidenceThresh  = 0.35;             % A threshold to determine if a track is true positive or false alarm by taking the recent average scores up to "timeWindowSize" frames.
option.ageThresh            = 10;               % A threshold to determine the minimum length required for a track being true positive.
option.visThresh            = 0.6;              % A threshold to determine the minimum visibility value for a track being true positive.

option.ImSize = 250;                            % A size of the input image (scan points to image), Be careful, it must be similer to the size of trained images. 
option.Ch = 3;                                  % A channel image size. set it 3 for RGB images, and 1 for Gray images.    
    
option.OverlapTh = 0.2;                         % A tuning parameter to specify the bounding box overlap ratio.                        
option.OnLineDetection = 0;                     % A logical parameter to switch between Online and offline detection.                                               
option.MorePallets = 1;                         % A logical parameter to switch between 1-pallet and 2-pallets dataset.

NumOfFrames = 40;                               % The number of frames.
nextId = 1;                                     % ID of the next track.   

% The function "TrainedCNNs" is used to load the pre-trained CNNs to detect and classify the ROIs.
[detector, ConvNet] = TrainedCNNs();

% The function "DeleteImages" deletes the old tracking results in the working folders (e.g. TrackingResults)
DeleteImages();

% The function "initializeTracks" creates an empty array of tracks.
tracks = initializeTracks();

for f = 1:NumOfFrames
    TestImage = readFrame();    
    [bboxes,scores,centroids,~] = detectPallets();
    predictNewLocationsOfPalletTracks();
    
    % The first method for Assigning Detections to Tracks (e.g. Bounding Boxes Overlaps)
    [assignments, unassignedTracks, unassignedDetections] = DetectionsAssignment();
    
    % The second method for Assigning Detections to Tracks (e.g. Measuring The Distance)
    % [assignments, unassignedTracks, unassignedDetections] = detectionToTrackAssignmentWithDist();
    
    updateAssignedTracks();
    updateUnassignedTracks();
    deleteLostTracks();
    createNewTracks();

    displayTrackingResults();
end

% Call "ImagesToVideo" for showing the final results
ImagesToVideo();

%============================================= Functions ========================================================>

% Use "TrainedCNNs" function to load the Pre-trained CNNs to detect and classify the ROIs.
function [detector, ConvNet] = TrainedCNNs()
    % Load the Pre-trained Fast R-CNN
    data = load('PreTrained_CNNs/TrainedFasterR-CNN/FastCNNPallet340RGBImages26.mat');
    detector = data.detector;

    % Load the Pre-trained CNN classifier.
    Net = load('PreTrained_CNNs/TrainedCNN/CNN_10Kfold_afterfold_450pallet.mat');
    ConvNet = Net.ConvNet;
end
%================================================================================================================>

% To delete all previous images in the folders
function DeleteImages()
    delete TrackingResults/StrongestRegions/*.png
    delete TrackingResults/ExtractedRegions/*.png
    delete TrackingResults/FinalRegions/*.png
    delete TrackingResults/MaxScore/*.png
    delete TrackingResults/Frames/*.png
    delete TrackingResults/Frames/Tracks/*.png
end

%================================================================================================================>
% The initializeTracks function creates an array of tracks which contains
% all information about the each track
function tracks = initializeTracks()
        % Create an empty array of tracks
        tracks = struct(...
            'id', {}, ...
            'color', {}, ...
            'bboxes', {}, ...
            'scores', {}, ...
            'kalmanFilter', {}, ...
            'age', {}, ...
            'totalVisibleCount', {}, ...
            'confidence', {}, ...
            'predPosition', {}, ...
            'consecutiveInvisibleCount', {});
end

%================================================================================================================>
% Use "readFrame" function to read the input images sequentially. 
% Moreover, you can call "ScanToImage" function to convert scan points to Images (only for a new data or an online detection)
function TestImage = readFrame()
    if ~option.OnLineDetection
        frameNo = 'Frame No. %d\n'; fprintf(frameNo,f);
        
        % Note: Generating all images have been done using "ScanpointToImage.m" script.
        if ~option.MorePallets
            % load the test image
            TestImage = imread(sprintf('TrackingResults/Images250Size/OnePalletDataset/ArtificialData/%d.png', f));
        else
            TestImage = imread(sprintf('TrackingResults/Images250Size/TwoPalletsDataset/ArtificialData/%d.png', f));
        end
    else
        % Be sure that you initialized ROS before running this code, using the the following command
        % rosinit;
        
        % Subscribe to the topic /scan.
        laser = rossubscriber('/scan');
        scan = receive(laser,10);
        % plot(scan);     xlim([0 5]);     ylim([-3 3]);
        % pause(1), clf('reset'); close;

        % Compute the Cartesian of the scanned data.
        data = readCartesian(scan);
        x = data(:,1);
        y = data(:,2);
        ScanToImage(); 
    end
        
    % Use "ScanToImage" function to convert the scan points to 2D images
    function ScanToImage()
        set(gcf,'color','white')
        c = linspace(1,15,length(x));
        h = scatter(y(f,:),x(f,:), c, 'filled');
        xlim([-3 3]);  ylim([0 5]);
        axis off
        saveas(h,sprintf('TrackingResults/%d.png',f));
        I = imread(sprintf('TrackingResults/%d.png', f));
        TImage = imresize(I,[option.ImSize option.ImSize]);
        imwrite(TImage,sprintf('TrackingResults/Images250Size/%d.png', f))
    end
end

%================================================================================================================>
% Use "detectPallets" function to detect the pallet based on the pre-trained Faster R-CNN detector and CNN classifier. 
% It returns:
%          1. The final bounding box (which represents the pallet class), 
%          2. The corresponding classification scores and their centriods,
%          3. Sort the entire detection results.   
function [bboxes,scores,centroids,Results] = detectPallets()
    
    % Run The Pre-trained Fast R-CNN detector 
    [AllBboxes, AllScores] = detect(detector,TestImage);
    
    % Save the detector results
    Results(f).Boxes = AllBboxes;
    Results(f).Scores = AllScores;
    % Results(i).Labels = labels;
    
    % Annotate detections in the image and save all images.
    I = insertObjectAnnotation(TestImage, 'rectangle', AllBboxes, AllScores);
    imwrite(I,sprintf('TrackingResults/AllRegions/%d.png', f))
    % figure; I = imresize(I, 1.5); imshow(I)
    
    % Display the max. score detected object
    [Maxscore, Idx] = max(AllScores);
    Maxbbox = AllBboxes(Idx, :);
    MaxImage = insertObjectAnnotation(TestImage, 'rectangle', Maxbbox, Maxscore);
    imwrite(MaxImage,sprintf('TrackingResults/MaxScore/%d.png', f))
    
    % Select the strongest detected objects
    [selectedBbox,selectedScore,selectedIdx] = selectStrongestBbox(AllBboxes,AllScores, ...
                                                             'RatioType', 'Min','OverlapThreshold', option.OverlapTh);
    % Save the strongest regions
    Results(f).StrongestBoxes = selectedBbox;
    Results(f).StrongestScores = selectedScore;
    Results(f).StrongestLabels = selectedIdx;

    % Annotate detections in the image and save all images.
    StrongestRegionsImage = insertObjectAnnotation(TestImage, 'rectangle', selectedBbox, selectedScore);
    imwrite(StrongestRegionsImage,sprintf('TrackingResults/StrongestRegions/%d.png', f))
    
    % Call "CNNClassifier" function in order to perform classifications using CNN
    [bboxes,scores,centroids] = CNNClassifier(); 
    
    % save the image with only the output of CNNCLassifier (pallet class)
    if ~isempty(bboxes)
        % save the image with only the output of CNNCLassifier (pallet class)
        FinalROIs = insertObjectAnnotation(TestImage, 'rectangle', bboxes, scores);
        imwrite(FinalROIs,sprintf('TrackingResults/FinalRegions/%d.png', f))
    else
        % In case of there is no detected ROIs.
        FinalMessage = 'No Detected Pallet in Image No: %d\n'; fprintf(FinalMessage,f);
        imwrite(TestImage,sprintf('TrackingResults/FinalRegions/%d.png', f))
    end
   
    function [bboxes,scores,centroids] = CNNClassifier()

        % Extract the strongest regions
        RegionBoxes = selectedBbox;
        RegionsScore = selectedScore;

        % Couter for Pallet regions
        Counter = 1; 
        % In order to save the final detected regions by CNNs
        FinalBboxes = []; FinalScores = []; 

        if ~isempty(RegionBoxes)
            for B = 1:size(RegionBoxes,1)
                Bbox = RegionBoxes(B,:);
                BboxScore = RegionsScore(B,:);

                % Call the function to extract the ROI from the input image
                OutputImage = ImageWithOnlyROI(Bbox, TestImage);
                whiteImage = OutputImage;
                imwrite(whiteImage,sprintf('TrackingResults/ExtractedRegions/%d_%d.png', f,B))

                % Convert 2D array into 4D array
                TestingData = {whiteImage};
                TestData4D = reshape(cat(3,TestingData{:}),option.ImSize,option.ImSize,option.Ch,length(TestingData));
                TestData4D = im2double(TestData4D);

                % Testing the strongest regions using CNN
                YTest = classify(ConvNet,TestData4D);

                % Check The Pallet Regions
                if YTest == categorical(1)
                    % Save all detected pallet bounding boxes 
                    FinalBboxes(Counter,:) = Bbox;
                    FinalScores(Counter,:) = BboxScore;
                    Counter = Counter + 1;
                end
            end
        end

        % Calculate the centroids of final ROIs
        if isempty(FinalBboxes)
            % Function outputs
            bboxes = [];
            scores = 0;
            centroids = [];
        else
            centroids = [(FinalBboxes(:, 1) + FinalBboxes(:, 3) / 2), ...
                        (FinalBboxes(:, 2) + FinalBboxes(:, 4) / 2)];
             % Function outputs
            bboxes = FinalBboxes;
            scores = FinalScores;       
        end
    end
end

%================================================================================================================>
% The "predictNewLocationsOfPalletTracks" function uses the motion-based Kalman filter 
% for predicting the position of the centroid of each track in the current frame.
function predictNewLocationsOfPalletTracks()
    for i = 1:length(tracks)
        bbox = tracks(i).bboxes(end, :);

        % Predict the current location of the track.
        predictedCentroid = predict(tracks(i).kalmanFilter);

        % Shift the bounding box so that its center is at the predicted location.
        predictedCentroid = predictedCentroid - bbox(3:4) / 2;
        tracks(i).predPosition = [predictedCentroid, bbox(3:4)];
    end
end

%================================================================================================================>
%  method No 1
% The DetectionsAssignment function is used for assigning the object detections to the
% existing tracks in the current frame by minimizing the cost function. This cost function
% is mainly computed based on the ratio of the overlap between the predicted and the de-
% tected bounding box.
 function [assignments, unassignedTracks, unassignedDetections] = DetectionsAssignment()

    % Compute the overlap ratio between the predicted boxes and the detected boxes, 
    % and compute the cost of assigning each detection to each track. 
    predBboxes = reshape([tracks(:).predPosition], 4, [])';
    cost = 1 - bboxOverlapRatio(predBboxes, bboxes);

    % 'costOfNonAssignment': Cost of not assigning detection to any track or track to detection.
    % Setting it too low increases the likelihood of creating a new track, and may result in track fragmentation.
    % Setting it too high may result in a single track corresponding to a series of separate moving objects.
    cost(cost > option.gatingThresh) = 1 + option.gatingCost;

    % Solve the assignment problem (the Munkres' version of the Hungarian algorithm).
    [assignments, unassignedTracks, unassignedDetections] = ...
        assignDetectionsToTracks(cost, option.costOfNonAssignment);
 end

%================================================================================================================>
% Assign Detections to Tracks --- method 2
function [assignments, unassignedTracks, unassignedDetections] = detectionToTrackAssignmentWithDist()

        nTracks = length(tracks);
        nDetections = size(centroids, 1);

        % Compute the cost of assigning each detection to each track.
        cost = zeros(nTracks, nDetections);
        for i = 1:nTracks
            cost(i, :) = distance(tracks(i).kalmanFilter, centroids);
        end

        % Solve the assignment problem.
        [assignments, unassignedTracks, unassignedDetections] = ...
            assignDetectionsToTracks(cost, option.costOfNonAssignment);
    end

%================================================================================================================>
% The "UpdateAssignedTracks" function updates the assigned tracks, each track with 
% the corresponding detections.
function updateAssignedTracks()
    numAssignedTracks = size(assignments, 1);
    for i = 1:numAssignedTracks
        trackIdx = assignments(i, 1);
        detectionIdx = assignments(i, 2);

        centroid = centroids(detectionIdx, :);
        bbox = bboxes(detectionIdx, :);

        % Correct the estimate of the object's location using the new detection.
        correct(tracks(trackIdx).kalmanFilter, centroid);

        % Stabilize the bounding box by taking the average of the size of recent (up to) 4 boxes on the track.
        T = min(size(tracks(trackIdx).bboxes,1), 4);
        w = mean([tracks(trackIdx).bboxes(end-T+1:end, 3); bbox(3)]);
        h = mean([tracks(trackIdx).bboxes(end-T+1:end, 4); bbox(4)]);
        if T == 4
            % Add 4 pixels to the height to be sure that the object detected completely
            % Note that: this is an optional step
            h = h + 4; w = w + 0.4;
        end
        tracks(trackIdx).bboxes(end+1, :) = [centroid - [w, h]/2, w, h];

        % Update track's age.
        tracks(trackIdx).age = tracks(trackIdx).age + 1;

        % Update track's score history
        tracks(trackIdx).scores = [tracks(trackIdx).scores; scores(detectionIdx)];

        % Update visibility.
        tracks(trackIdx).totalVisibleCount = tracks(trackIdx).totalVisibleCount + 1;
        tracks(trackIdx).consecutiveInvisibleCount = 0;
        
        % Adjust track confidence score based on the maximum detection score in the past 'timeWindowSize' frames.
        T = min(option.timeWindowSize, length(tracks(trackIdx).scores));
        score = tracks(trackIdx).scores(end-T+1:end);
        tracks(trackIdx).confidence = [max(score), mean(score)];
    end
end

%================================================================================================================>
% The "UpdateUnassignedTracks function" marks each unassigned track as an invisible by
% setting the confidence score to 0.
function updateUnassignedTracks()
    for i = 1:length(unassignedTracks)
        idx = unassignedTracks(i);
        tracks(idx).age = tracks(idx).age + 1;
        tracks(idx).bboxes = [tracks(idx).bboxes; tracks(idx).predPosition];
        tracks(idx).scores = [tracks(idx).scores; 0];
        tracks(idx).consecutiveInvisibleCount = tracks(idx).consecutiveInvisibleCount + 1;

        % Adjust track confidence score based on the maximum detection score in the past 'timeWindowSize' frames
        T = min(option.timeWindowSize, length(tracks(idx).scores));
        score = tracks(idx).scores(end-T+1:end);
        tracks(idx).confidence = [max(score), mean(score)];
    end
end

%================================================================================================================>
% The "DeleteTracks" function deletes the created tracks that have not been visible for too
% many successive frames or have a total number of detected frames ( totalVisibleCount )
% less than a predefined value (visThresh).
function deleteLostTracks()
        if isempty(tracks)
            return;
        end

        % Compute the fraction of the track's age for which it was visible.
        ages = [tracks(:).age]';
        totalVisibleCounts = [tracks(:).totalVisibleCount]';
        visibility = totalVisibleCounts ./ ages;

        % Check the maximum detection confidence score.
        confidence = reshape([tracks(:).confidence], 2, [])';
        maxConfidence = confidence(:, 1);
        AvgConfidence = confidence(:, 2);
        
        % Find the indices of 'lost' tracks.
        lostInds = (ages <= option.ageThresh & visibility <= option.visThresh) | ...
             (maxConfidence <= option.confidenceThresh | ...
             AvgConfidence <= option.MinconfidenceThresh);

        % Delete lost tracks.
        tracks = tracks(~lostInds);
end

%================================================================================================================>
% The CreateNewTracks function creates new tracks from the unassigned detections. 
% Assuming that any unassigned detection is a start of a new track.
function createNewTracks()
    unassignedCentroids = centroids(unassignedDetections, :);
    unassignedBboxes = bboxes(unassignedDetections, :);
    unassignedScores = scores(unassignedDetections);

    for i = 1:size(unassignedBboxes, 1)
        centroid = unassignedCentroids(i,:);
        bbox = unassignedBboxes(i, :);
        score = unassignedScores(i);

        % Create a Kalman filter for object tracking.
        kalmanFilter = configureKalmanFilter('ConstantVelocity', centroid, [2, 1], [5, 5], 100);

        % Create a new track.
        newTrack = struct(...
            'id', nextId, ...
            'color', 255*rand(1,3), ...
            'bboxes', bbox, ...
            'scores', score, ...
            'kalmanFilter', kalmanFilter, ...
            'age', 1, ...
            'totalVisibleCount', 1, ...
            'confidence', [score, score], ...
            'predPosition', bbox, ...
            'consecutiveInvisibleCount', 0);

        % Add it to the array of tracks.
        tracks(end + 1) = newTrack; %#ok<AGROW>

        % Increment the next id.
        nextId = nextId + 1;
    end
end

%================================================================================================================>
% The "DisplayTrackingResults" function is used for display purpose.It draws a colored
% bounding box and label ID for each track. It displays only the reliable tracks which
% have an average detection confidence score greater than a predefined threshold or their
% ages greater than half of the minimum length required for considering a track as a True Positive (TP). 
function displayTrackingResults()
    
    frame = TestImage;
    
    if ~isempty(tracks)    
        ages = [tracks(:).age]';
        confidence = reshape([tracks(:).confidence], 2, [])';
        maxConfidence = confidence(:, 1);
        avgConfidence = confidence(:, 2);
        opacity = min(0.5,max(0.1,avgConfidence/3));
        % Display only the reliable objects    
        DispInds = (ages > option.ageThresh & maxConfidence > option.confidenceThresh) | ...
                     (ages > option.ageThresh / 2);
       
        for k = 1:length(tracks)
            if DispInds(k)

                % scale bounding boxes for display
                bb = tracks(k).bboxes(end, :);
                id = tracks(k).id;
                annotation = sprintf('T%d: %.4g', id, avgConfidence(k));
                frame = insertShape(frame, ...
                                        'FilledRectangle', bb, ...
                                        'Color', tracks(k).color, ...
                                        'Opacity', opacity(k));
                frame = insertObjectAnnotation(frame, ...
                                        'rectangle', bb, ...
                                        annotation, ...
                                        'Color', tracks(k).color);
                                    
                imwrite(frame,sprintf('TrackingResults/Frames/Tracks/%d.png', f))                    
            else
                bb = tracks(k).bboxes(end, :);
                score = tracks(k).scores(end);
                frame = insertObjectAnnotation(frame, 'rectangle', bb, score);
                imwrite(frame,sprintf('TrackingResults/Frames/Tracks/%d.png', f))
            end
        end
    end
end

%================================================================================================================>
% use "ImagesToVideo" function to convert the sequential images into video
function ImagesToVideo()
    F = [];
    outputVideo = VideoWriter('PalletTrackingOutput.avi');
    open(outputVideo)
    for i = 1:NumOfFrames
        % load the test image
        OutputImage = imread(sprintf('TrackingResults/Frames/Tracks/%d.png', i));
        %F(i) = im2frame(TestImage);
        writeVideo(outputVideo,OutputImage)
    end
    close(outputVideo)
    implay('PalletTrackingOutput.avi',1)
end

% END of the main function "PalletTrackingusingKalmanFilter"
end

