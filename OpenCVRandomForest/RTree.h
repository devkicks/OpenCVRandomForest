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
#include "LeafNode.h"

#define C_NUM_FEATURES 1
#define C_THRESHOLD_IT 10

// helper for displaying the tree structure
#define C_NUM_HORIZONTAL_SPACE 200 //pixels
#define C_NUM_VERTICAL_SPACE 200 // pixels
#define C_PADDING_DISPLAY 500 // for putting the tree in the middle of a padded image

// TODO:: Add verbrose mode for debugging

class CRTree
{
private:
	// Temporary tree table to store treetable while Training
	std::vector<cv::Mat> m_tempTable;
	
	// Treetable for storing tree structure
	cv::Mat m_treetable;

	// Leaftable for storing leaf nodes
	std::vector<CLeafNode> m_leaves;

	// Learning params

	// minimum samples required to split
	unsigned int m_minSamples;

	// maximum depth of the tree
	unsigned int m_maxDepth;

	// size of each split node
	unsigned int m_sizeOneNode;

	// for displaying tree structure
	cv::Mat displayTree;
	std::stringstream stringStr;
	cv::Scalar treeColor;

public:
	// constructor
	CRTree(int minSamples=10, int maxDepth=8);
	
	// destructor
	~CRTree(void);

	// Predict: Infer a leafnode for an input data
	CLeafNode Predict(const cv::Mat &inFeatures) const;

	// Train: Tree training using n samples from training set
	void Train(const cv::Mat &inData, const cv::Mat &inLabel);

	// functions for helping display the decision tree structure
	void displayDecisionTree(int idxImage);

private:
	// grow: Recursive function for growing trees
	void grow(const cv::Mat &inData, const cv::Mat &inLabel, unsigned int node, unsigned int depth);

	// makeLeaf: Creating leaves at terminal nodes
	void makeLeaf(const cv::Mat& inData, const cv::Mat& inLabel);

	// optimizeTest: Extremely randomized tree optimization
	bool optimizeTest(cv::Mat &partitionMap, const  cv::Mat &inData, const cv::Mat &inAngle, unsigned iter, cv::Mat &bestThreshold);
	
	// generateTest: Generate test i.e. randomly selected dimension of input features
	void generateTest(int* test, unsigned int lengthVec);

	// evaluateTest: Evaluate a random candidate split
	void evaluateTest(const cv::Mat &inData, cv::Mat &retFeatures, int* test);
	
	// split: Partition data using a given split criteria - returns hashmap called partitionMap
	//		  that indicates left and right partitions
	void split(cv::Mat &partitionMap, const cv::Mat &selFeatures, const cv::Mat &cThresholds);

	// measureInformationGain: Information gain energy calculator 
	//          - takes partitionMap returned from split() and the corresponding target labels
	double measureInformationGain(const cv::Mat &inLabel, const cv::Mat &partitionMap);
	
	// functions for helping display the decision tree structure
	void displayDecisionTree(cv::Vec2i curPts, int realWidth, int cNode, int cDepth);


	//.... isnan and isinf temprorary implementation
	int isnan(double x) { return x != x; }
	int isinf(double x) { return !isnan(x) && isnan(x - x); }
};

