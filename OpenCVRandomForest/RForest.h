/* 
Multi-variate Random Forest code written in C++
_______________________________________________

This code contains a framework for multi-variate Random Forest regression.
Written by Dr Muhammad Asad, City, University of London
<masadcv@gmail.com>

Source code drawn from a number of sources and examples, including contributions from
- Pierre-Luc Bacon <plbacon@cim.mcgill.ca> https://github.com/pierrelux/rf_pose
- Juergen Gall, BIWI, ETH Zurich <gall@vision.ee.ethz.ch>
and others

For educational/research use

Please cite one of the following work(s) when using this code for research

-"Learning Marginalization through Regression for Hand Orientation Inference",
  Muhammad Asad, Greg Slabaugh, 
  Computer Vision and Pattern Recognition (CVPR) Second Workshop on Observing and Understanding Hands in Action (HANDS) 2016.

-"Hand Orientation Regression using Random Forest for Augmented Reality", 
  Muhammad Asad, Greg Slabaugh, 
  First International Conference on Augmented and Virtual Reality (AVR) 2014. 


Potential ways to modify the code:  
- Add a prediction model, e.g. Kernel Density Estimation 
- Improve debugging options 
- Try a different Energy function
- use OMP to parallelize the training
- make the tree options available to user; currently these options are encapsulated by CRForest class

Written by 
Dr. Muhammad Asad <masadcv@gmail.com>
City, University of London
*/

#pragma once
#include "RTree.h"

// TODO:: Add verbrose mode for debugging
// TODO:: Add OMP support for parallel training
class CRForest
{
private:
	// Forest model - collection of Extremely Randomized Trees
	CRTree *decisionForest;

	// number of trees in the forest
	unsigned int m_numOfTrees;
	
public:
	// constructor
	CRForest(unsigned int numOfTrees = 5);

	// destructor
	~CRForest(void);

	// Train: Training Random Forest using input data and regressors params
	void Train(const cv::Mat &inData, const cv::Mat &inLabel, const double& minVal = 0.0f, const double& maxVal = 3.0f);

	// Predict: Infer the target, given an input feature
	cv::Mat Predict(const cv::Mat &inData);

	// Function for displaying structure of a specific tree
	void displayTree(int treeIdx)
	{
		if(treeIdx < m_numOfTrees)
			decisionForest[treeIdx].displayDecisionTree(treeIdx);
	}
	
private:
	// bagging function - randomly select data for training trees
	cv::vector<cv::Mat> baggingFunction(const cv::Mat &inData, const cv::Mat &inLabel, float ratio = 0.8f);

	// bagging function with replacement
	cv::vector<cv::Mat> baggingFunctionReplacement(const cv::Mat &inData, const cv::Mat &inLabel, float ratio = 0.8f);

	// randSample: returns indices of randomly sampled samples
	// from a total numOfData - extracts numOfSamples
	// without replacement 
	cv::vector<int> randSample(int numOfData, int numOfSamples); 

	// with replacement
	cv::vector<int> randSampleReplacement(int numOfData, int numOfSamples); 
};

