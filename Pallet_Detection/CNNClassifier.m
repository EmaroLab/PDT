function [bboxes,scores,centroids] = CNNClassifier()
%%
 % This fuction is used to classify the detected regions by the pre-trained "Faster R-CNN" detector. 
 % It calls the pre-trained CNN Classifier in order to perform classifications.
 % Function Output:
 %              bboxes: It represents the detected regions which classified
 %                      as Class 1 (where the pallet is present)by the pre-trained CNN.
 %              scores: The corresponding confidence scores.
 %           centroids: It represents the center point of the bounding boxes. 
 %                      It is a N-by-2 matrix, where each row has a form of [x, y].
 % Function inputs:
 %           TestImage: The 2D input image of the scene.                 
 %        selectedBbox: It refers to the strongest detected objects by the
 %                      Pre-trained Faster R-CNN
 %       selectedScore: The corresponding confidence scores.
 %%
 
% Extract the strongest regions
RegionBoxes = selectedBbox;
RegionsScore = selectedScore;

% Counter for Pallet regions
Counter = 1; 
% In order to save the final detected regions
FinalBboxes = []; FinalScores = [];

if ~isempty(RegionBoxes)
    for B = 1:size(RegionBoxes,1)

        Bbox = RegionBoxes(B,:);
        BboxScore = RegionsScore(B,:);

        % Call the function "ImageWithOnlyROI" to extract only the ROI from
        % the input image with the same size of the input image.
        OutputImage = ImageWithOnlyROI(Bbox, TestImage);
        whiteImage = OutputImage;
        imwrite(whiteImage,sprintf('TrackingResults/ExtractedRegions/%d_%d.png', i,B))

        % Convert 2D array into 4D array
        TestingData = {whiteImage};
        TestData4D = reshape(cat(3,TestingData{:}),option.ImSize,option.ImSize,option.Ch,length(TestingData));
        TestData4D = im2double(TestData4D);

        % Testing the strongest regions using CNN
        YTest = classify(ConvNet,TestData4D);

        % Check The Pallet Regions
        if YTest == categorical(1)
            % Sort all detected pallet bounding boxes 
            FinalBboxes(Counter,:) = Bbox;
            FinalScores(Counter,:) = BboxScore;
            Counter = Counter + 1;
        end
    end
end

% Calculate the centroids of all regions belongs to pallet class.
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
