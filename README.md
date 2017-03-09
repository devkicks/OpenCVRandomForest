OpenCVRandomForest: Implementation of Multi-variate Random Forest using OpenCV

OpenCVRandomForest
============

Code provides basic framework for multi-variate Random Forest regression.

Pre-requisites
==============

- OpenCV 2.x (Included in repo)

Usage
=====
The models assume that the data is arranged in columns i.e. each column has a new sample in data, such as:
                  
                  trainFeatures = [x1, x2, x3, ......, xN],
                  trainTarget   = [y1, y2, y3, ......, yN],
where xi are input feature vectors and yi are multi-variate continuous target labels

Initializing a Forest structure with a specified number of trees (numOfTrees)
                  
                  int numOfTrees = 100;
                  CRForest myForest(numOfTrees); // Random Foreset with 100 Trees
                  
Training:
                  
                  myForest.Train(trainFeatures, trainTarget);
                  // Training with features and multi-variate continuous target labels

Prediction:
Given a single feature vector (column vector), the trained model can be used to infer the target as:

                  cv::Mat predTarget = myForest.Predict(testFeature);

where predTarget is a multi-variate column vector with same dimension as a single target label in the training set.

This repository also contains multi-variate regression datasets from UCI Machine Learning Repository. Also included are sample Matlab script files for analyzing the output of the model.

Visualizing Tree Structure
==========================

The code include a small utility function for visualizing tree structures.

<img src="https://github.com/devkicks/OpenCVRandomForest/blob/master/OpenCVRandomForest/DisplayTrees/smallImage/imageGif.gif" alt="Color Image" width="400"/>


Citation
========

For educational/research use

Please cite one of the following work(s) when using this code for research

-"Learning Marginalization through Regression for Hand Orientation Inference",
  Muhammad Asad, Greg Slabaugh, 
  Computer Vision and Pattern Recognition (CVPR) Second Workshop on Observing and Understanding Hands in Action (HANDS) 2016.

-"Hand Orientation Regression using Random Forest for Augmented Reality", 
  Muhammad Asad, Greg Slabaugh, 
  First International Conference on Augmented and Virtual Reality (AVR) 2014. 

Future Work
===========

Possible extension of this code include:
- Add a prediction model, e.g. Kernel Density Estimation 
- Improve debugging options 
- Try a different Energy function
- use OMP to parallelize the training
- make the tree options available to user; currently these options are encapsulated by CRForest class


Copyright (C) 2017  Muhammad Asad
- Contact: masadcv@gmail.com


This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.


This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.


You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
