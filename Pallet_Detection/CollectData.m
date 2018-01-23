% This script can be used for acquiring data from the laser and convert it
% from Polar coordinates to Cartesian coordinates.
% This script is used for collecting data while the robot is moving towards
% the pallet. We considered 3 paths for the same pallet pose. 

% The collected data of each path is saved in "AllData/Trajectory/PalletPositionPaths1.mat..... PalletPositionPaths4.mat" 

clc, clear variables, close all;

% Initialize ROS Network
rosinit
laser = rossubscriber('/scan');

PathsNo = 3;           %  # of paths to approach the pallet.
PointsPerpath = 10;    %  # of frames aquired while the robot moving towards the pallet.
ScanPoints = 761;      %  There total of scan points given by the laser for each frame.     

ScanRanges = zeros(PathsNo*PointsPerpath,ScanPoints);
x = zeros(PathsNo*PointsPerpath,ScanPoints);
y = zeros(PathsNo*PointsPerpath,ScanPoints);

for PathNo = 1:PathsNo
    Path = 'Path No: %d\n'; 
    fprintf(Path,PathNo)
    for i = 1:PointsPerpath
        % wait for button press
        k = waitforbuttonpress;
        Point = 'Path No: %d, Point No: %d\n';
        fprintf(Point,PathNo, i)
        
        scan = receive(laser,10);
        plot(scan);    xlim([-1 5]);     ylim([-5 5]);
        pause(1), clf('reset'); close;
        Ranges = scan.Ranges';
        ScanRanges(i+PointsPerpath*(PathNo-1),:) = Ranges;
        
        dataxy = readCartesian(scan);
        x(i+PointsPerpath*(PathNo-1),:) = dataxy(:,1);
        y(i+PointsPerpath*(PathNo-1),:) = dataxy(:,2);
    end
end