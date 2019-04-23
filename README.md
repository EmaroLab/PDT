# PDT - Pallets Detector and Tracker

## What is PDT?
 
**PDT** is a public repository of codes and *datasets* for detecting and tracking the [pallets](https://en.wikipedia.org/wiki/EUR-pallet) in Warehouses based on machine learning approaches using a 2D Laser Rangefinder (LRF). 

The proposed system for pallets detection and tracking is mainly composed of two main components:
1. a Faster Region-based Convolutional Network (Faster R-CNN) detector which is followed by CNN classifier for detecting the pallets.
2. a motion-based Kalman filter for tracking and increasing the confidence of the presence of the pallet.

## Licensing and Citations 

This dataset is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY, including the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. The authors allow the users of the Data Set to use and modify it for their own research. Any commercial application, redistribution, etc... has to be arranged between users and authors individually.

Eventually, if you use PDT or relevant to your academic work, please cite:

@article{mohamed2018detection,<br/>
title={[Detection, localisation and tracking of pallets using machine learning techniques and 2D range data](https://arxiv.org/abs/1803.11254)},<br/>
author={Mohamed, Ihab S. and Capitanelli, Alessio and Mastrogiovanni, Fulvio and Rovetta, Stefano and Zaccaria, Renato},<br/>
journal={arXiv preprint arXiv:1803.11254},<br/>
year={2018}<br/>
}<br/>

@article{mohamed20182d,<br/>
  title={[A 2D laser rangefinder scans dataset of standard EUR pallets](https://www.sciencedirect.com/science/article/pii/S235234091930188X?via%3Dihub)},<br/>
  author={Mohamed, Ihab S and Capitanelli, Alessio and Mastrogiovanni, Fulvio and Rovetta, Stefano and Zaccaria, Renato},<br/>
  journal={arXiv preprint arXiv:1805.08564},<br/>
  year={2018}<br/>
}<br/>

> For further license information, please contact the authors.

## Errata Corrige

In the paper "[A 2D laser rangefinder scans dataset of standard EUR pallets](https://www.sciencedirect.com/science/article/pii/S235234091930188X?via%3Dihub)" <sup>[1](#preprintRef)</sup> we stated that we provide 4 continous trajectories for testing purposes. Indeed, the files you will find in the `PDT/Pallet_Detection/AllData/` folder are not continous, but they contain 3 separate trajectories, each counting 40 frames and about *4m* long. For each set, one trajectory is taken while approaching directly the pallet, while the other two are taken trying to keep the pallet on the right and on the left side of the sensor respectively. Each set differs by the sensor starting position, the pallet's position and orientation, and obstacles' disposition and dimensions.

## Authors contacts

If you want to be informed about dataset updates and new code releases, obtain further information about the provided dataset, or contribute to its development please write to:

- **Ihab S. Mohamed**, dept. DIBRIS Università degli Studi di Genova (Italy), ihab.s.mohamed@gmail.com
- **Alessio Capitanelli**, dept. DIBRIS Università degli Studi di Genova (Italy), alessio.capitanelli@dibris.unige.it
- **Fulvio Mastrogiovanni**, dept. DIBRIS Università degli Studi di Genova (Italy), fulvio.mastrogiovanni@unige.it
- **Stefano Rovetta**, dept. DIBRIS Università degli Studi di Genova (Italy), stefano.rovetta@unige.it
- **Renato Zaccaria**, dept. DIBRIS Università degli Studi di Genova (Italy), renato.zaccaria@unige.it

<a name="preprintRef">1</a> - Preprint version available on [arXiv](https://arxiv.org/abs/1805.08564)
