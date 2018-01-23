function OutputImage = ImageWithOnlyROI(Bbox, TestImage)
% This function is used to generate a 2D image with only the ROIs. 
% The generated image has the same size as the original image.

    croppedImage = imcrop(TestImage, Bbox);
    % imshow(croppedImage)

    % Image with only interested points 
    whiteImage = 255 * ones(250, 250, 3, 'uint8');
    % Check the Image borders
    Width = Bbox(2)+Bbox(4); 
    Height = Bbox(1)+Bbox(3);
    if Width>250
        Width = 250;
    end
    if Height>250
        Height =250;
    end
    whiteImage(Bbox(2):Width,Bbox(1):Height,:) = croppedImage;
    OutputImage = whiteImage;
end