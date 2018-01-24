# *Folders Description*

* `AllData`: contains the main part of the dataset which has been used for training and testing the proposed detection system. It contains the raw data which has been acquired from the LRF (**see Subfolders**: `Class1 & Class2`) and the generated 2D images (**see Subfolder**: `PalletImages`). 	 It also contains the acquired data while the robot is moving towards the Pallet (**see Subfolder**: `TrajectoryDataset`). 

* `PreTrained_CNNs`: contains the results of training the Faster R-CNN detector and the CNN classifier. These Pre-trained CNNs will be called by the Tracker in order to perform pallets tracking task.  

* `TrackingResults`: contains the results of the proposed tracking algorithm frame by frame. It also contains the dataset which has been used to check the robustness of our approach in the case of having one pallet or 2-pallets (**see Subfolder**: `Images250Size`).

# *Functions Description*

PDT is mainly composed of three sequential Phases:

1. `Phase #1` is mainly used to convert the acquired from laser scanner (raw data) into a 2D image (in both cases offline and on-line detection and 			tracking) in order to train the proposed Networks.

 
	There are two functions for performing this phase:

	1. `XYLaserScan.m`: matlab function that shows how to convert the dataset acquired from Polar to Cartesian coordinates in order to train 				the CNNs. All the acquired data have been saved in "AllData" Folder.
	2. `CollectData.m`: matlab function that is used for collecting data while the robot is moving towards the pallet in order to perform 			Pallet tracking. We considered 3 paths for the same pallet pose in each "Path" there are 10 acquired frames.

	
2. `Phase #2` is the phase of the training and testing the Faster R-CNN detector and the CNN classifier. It takes as input the created 2D image. 

	There are two main functions to fine-tune The proposed CNNs:
	
	1. `FasterRCNNPallet.m`: matlab function that shows how to train and test the Fast R-CNN detector using the ready data which we collected 		manually using "`Image Labeler App`". It also shows how to extract ROIs from the given image to train the CNN classifier. 
	- [x] Note that: this script is compatible with Matlab 2017 only (`e.g. R2017b`).
	- [] This is invalid for the rest of scripts.

	2. `CNNPalletClassification_WithValidation.m`: matlab function that is used for training the dataset using the CNN classifier in order to 			classify Pallet among other negatives tracks with high confidence scores. It also applies the k-fold cross-validation to 				evaluate the performance of the proposed system.


3. `Phase #3` is the tracking phase. Once the proposed CNNs is fine-tuned and tested, the tracking phase must be executed with the aim of
	detecting, recognizing and tracking the positive tracks (pallets) among other negatives tracks (false alarms).

	
	* `PalletTrackingUsingKalmanFilter.m`: This is the main function that is used to implement the pallets tracking Algorithm. This function in 			turn calls a number of sub-functions, each with a specific task. These functions must be called for each frame in order to 			perform the pallets tracking task.



