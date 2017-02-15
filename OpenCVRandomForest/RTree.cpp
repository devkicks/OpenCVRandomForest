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

#include "RTree.h"

// TODO:: Add verbrose mode for debugging

// constructor
CRTree::CRTree(int minSamples, int maxDepth)
{
	// using constructor to set Tree params
	m_minSamples = minSamples;
	m_maxDepth = maxDepth;
	//m_numLeaves = 0;

	// determine the size of each split
	m_sizeOneNode = 1 + C_NUM_FEATURES*2 + 2;
				//  1 variable to store the status + C_NUM_FEATURES*2 for storing the learned splits
				//				+ 2 for storing the index of the left and right subtree
}

// destructor
CRTree::~CRTree(void)
{	
	// delete [] m_leaves;

	// TODO:  release memory used by m_tempTreetable
}

// Predict: Infer a leafnode for an input data
CLeafNode CRTree::Predict(const cv::Mat &inFeatures) const
{
	// starting index for treetable
	int idxNode = 0;
	
	while(m_treetable.at<float>(0, idxNode)==-1)
	{
		bool isLeft = true;
		
		// apply learned tests
		for(int i = 0; i < C_NUM_FEATURES; i++)
		{
			isLeft &= (inFeatures.at<float>(m_treetable.at<float>(i+1, idxNode), 0) < m_treetable.at<float>(i+C_NUM_FEATURES+1,idxNode));
		}
		
		if(isLeft)
		{
			// go to left subtree
			idxNode = m_treetable.at<float>(1 + C_NUM_FEATURES*2, idxNode);
		}
		else
		{
			//go to right subtree
			idxNode = m_treetable.at<float>(1 + C_NUM_FEATURES*2 + 1, idxNode);
		}
	}
	return m_leaves[(int)m_treetable.at<float>(0, idxNode)];
}


// Train: Tree training using n samples from training set
void CRTree::Train(const cv::Mat &inData, const cv::Mat &inLabel)
{
	// change the seed to induce randomness
	cv::theRNG().state = time(NULL);

	std::cout << "Growing tree ..." << std::endl;

	// assuming the data is arranged in columns with fixed number of rows
	// each new sample is in the column with corresponding labels

	grow(inData, inLabel, 0, 0);

	// convert the temp Mat representation to permanent Mat
	// converting std::vector to a mat using horizontal concatenation
	// using hconcate function
	cv::hconcat(this->m_tempTable, this->m_treetable);

	// release memory from tempTable
	m_tempTable.clear();

	//std::cout << m_treetable << std::endl;
}

// grow: Recursive function for growing trees
void CRTree::grow(const cv::Mat &inData, const cv::Mat &inLabel, unsigned int node, unsigned int depth)
{
	int samples = inLabel.cols;
	
	cv::Mat tempTableEntry = cv::Mat::zeros(this->m_sizeOneNode, 1, CV_32FC1);

	if(depth >= m_maxDepth && inData.cols)
	{
		std::cout << "Reached maximum depth. Creating leaf" << std::endl;
		makeLeaf(inData, inLabel);
		return;
	}

	cv::Mat leftPartition, rightPartition, leftPartitionLabel, rightPartitionLabel;
	cv::Mat bestThreshold;
	cv::Mat partitionMap;
	partitionMap = cv::Mat::zeros(1, inData.cols, CV_32FC1);

	if(optimizeTest(partitionMap, inData, inLabel, samples, bestThreshold))
	{
		// OLD: m_treetable.at<float>(0 ,  node) = -1;
		tempTableEntry.at<float>(0, 0) = -1; // storing split node

		// TODO:: check this ++t is logically correct
		for(int t = 0; t < C_NUM_FEATURES*2; ++t)
		{
			// OLD: m_treetable.at<float>(t+1, node) = bestThreshold.at<float>(0,t);
			tempTableEntry.at<float>(t + 1, 0) = bestThreshold.at<float>(0, t);
		}

		// get size of the partition
		cv::Mat tempSum;
		cv::reduce(partitionMap, tempSum, 1 /*means reduced to single column*/ , CV_REDUCE_SUM);

		int numOfRight = tempSum.at<float>(0,0);
		int numOfLeft = inData.cols - numOfRight;

		// create containersw to store left and righ partition
		leftPartition = cv::Mat::zeros(inData.rows, numOfLeft, CV_32FC1);
		rightPartition = cv::Mat::zeros(inData.rows, numOfRight, CV_32FC1);

		leftPartitionLabel = cv::Mat::zeros(inLabel.rows, numOfLeft, CV_32FC1);
		rightPartitionLabel = cv::Mat::zeros(inLabel.rows, numOfRight, CV_32FC1);
		
		int lIdx = 0, rIdx = 0;

		// prepare the partitioned data for the next depth
		for(int i = 0; i < inData.cols; i++)
		{
			if(partitionMap.at<float>(0, i) == 0)
			{
				inData.col(i).copyTo(leftPartition.col(lIdx));
				inLabel.col(i).copyTo(leftPartitionLabel.col(lIdx++));
			}
			else
			{
				inData.col(i).copyTo(rightPartition.col(rIdx));
				inLabel.col(i).copyTo(rightPartitionLabel.col(rIdx++));
			}
		}

		m_tempTable.push_back(tempTableEntry.clone());
		tempTableEntry.release();

		// grow the tree in the depth, based on min samples
		if(leftPartition.cols > m_minSamples)
		{
			// std::cout << "Growing left branch" << std::endl;
			m_tempTable[node].at<float>(1 + C_NUM_FEATURES*2, 0) = m_tempTable.size();
			grow(leftPartition, leftPartitionLabel, m_tempTable.size(), depth + 1);
		}
		else
		{
			////std::cout << "Making leaf in left branch" << std::endl;
			m_tempTable[node].at<float>(1 + C_NUM_FEATURES*2, 0) = m_tempTable.size();
			makeLeaf(leftPartition, leftPartitionLabel);
		}

		// same for right
		if(rightPartition.cols > m_minSamples)
		{
			////std::cout << "Growing right branch" << std::endl;
			m_tempTable[node].at<float>(1 + C_NUM_FEATURES*2 + 1, 0) = m_tempTable.size();
			grow(rightPartition, rightPartitionLabel, m_tempTable.size(), depth + 1);
		}
		else
		{
			////std::cout << "Making leaf in left branch" << std::endl;
			m_tempTable[node].at<float>(1 + C_NUM_FEATURES*2 + 1, 0) = m_tempTable.size();
			makeLeaf(rightPartition, rightPartitionLabel);
		}
	}
	else
	{
		////std::cerr << "********** Could not find valid split. Making leaf" << std::endl;
		// TODO: Check if this is all right
		makeLeaf(inData, inLabel);
	}

}

// Create leaf node
void CRTree::makeLeaf(const cv::Mat& inData, const cv::Mat& inLabel) 
{
	////std::cout << "Making leaf " << m_numLeaves << " with " << inData.cols << " samples." << std::endl;

	// makeleaf assumes that the last node in tempTable is corresponding to leaf node
	// therefore it always adds a single entry
	cv::Mat tempTableEntry = cv::Mat::zeros(this->m_sizeOneNode, 1, CV_32FC1);
	m_tempTable.push_back(tempTableEntry.clone());
	tempTableEntry.release();

	// store leaf address
	//OLD: m_treetable.at<float>(0, node) = m_numLeaves;
	m_tempTable[m_tempTable.size()-1].at<float>(0, 0) = m_leaves.size();
	CLeafNode leaf;

	// Store sigma and mu
	if (inData.cols > 0) {
		leaf.createLeafWithData(inLabel);
	}
	
	//TODO:: Check and make a copy constructor that clones the individual Mats
	m_leaves.push_back(leaf);

	// Increase leaf counter
	//m_numLeaves += 1;
}

// optimizeTest: Extremely randomized tree optimization
bool CRTree::optimizeTest(cv::Mat &partitionMap, const  cv::Mat &inData, const cv::Mat &inAngle, unsigned iter, cv::Mat &bestThreshold)
{   	        
	cv::Mat returnPartitionMap;
	double bestSplit = -DBL_MAX;
	bool ret = false;
	int bestTest[C_NUM_FEATURES];
	// Find best test
	for(unsigned i = 0; i < 10; ++i) {    
		// generate binary test for pixel locations m1 and m2
		int tmpTest[C_NUM_FEATURES];
		cv::Mat retFeatures, maxVals, minVals;
		generateTest(tmpTest, inData.rows);

		// compute the test - i.e. get vals for a given dimension from all samples
		evaluateTest(inData, retFeatures, tmpTest);

		cv::reduce(retFeatures, maxVals, 1 /*means reduced to single column*/ , CV_REDUCE_MAX);
		cv::reduce(retFeatures, minVals, 1 /*means reduced to single column*/ , CV_REDUCE_MIN);

		// using axis aligned weak learner
		
		// basic check to see if there is variation in the data - i.e. if maxVal != minVal
		bool checkValsRange = true;
		for(int cI = 0; cI < C_NUM_FEATURES; cI++)
		{
			checkValsRange &= ((maxVals.at<float>(cI,0) - minVals.at<float>(cI,0)) > 0);
		}
		
		
		if(checkValsRange)
		{
			// Find best threshold
			for(unsigned int j = 0; j < C_THRESHOLD_IT; j++) { 
				// Generate some random thresholds
				cv::Mat tempThreshold = cv::Mat::zeros(maxVals.size(), maxVals.type());
				partitionMap = cv::Mat::zeros(1, inData.cols, CV_32FC1);
				for(int jj = 0; jj < tempThreshold.rows; jj++)
				{
					// randomly pick a threshold for each dimension -> row in data
					tempThreshold.at<float>(jj,0)= cv::theRNG().uniform(minVals.at<float>(jj,0), maxVals.at<float>(jj,0));
				}

				// Split training data into two sets A and B accroding to threshold 
				split(partitionMap, retFeatures, tempThreshold);

				cv::Mat checkSum;
				cv::reduce(partitionMap, checkSum, 1 /*means reduced to single column*/ , CV_REDUCE_SUM);

				// Do not allow empty set split
				if(checkSum.at<float>(0,0) > 5 && (partitionMap.cols -checkSum.at<float>(0,0)) > 5) {
					// Measure quality of split
					double score = measureInformationGain(inAngle, partitionMap);

					// Take binary test with best split
					if(score > bestSplit) {
						ret = true;
						bestSplit = score;
						returnPartitionMap = partitionMap.clone();
						memcpy(bestTest, tmpTest, sizeof(tmpTest));
						bestThreshold = tempThreshold.clone();
					}
				}
			}
		}
	}

	partitionMap = returnPartitionMap.clone();
	cv::Mat tempCpy = bestThreshold.clone();
	bestThreshold = cv::Mat::zeros(1, C_NUM_FEATURES*2, CV_32FC1);

	// make the return matrix
	if(ret)
	{
 	for(int i = 0; i < C_NUM_FEATURES*2; i++)
	{
		if(i < C_NUM_FEATURES)
		{
			bestThreshold.at<float>(0,i) = bestTest[i];
		}
		else
		{
			bestThreshold.at<float>(0,i) = tempCpy.at<float>(i-C_NUM_FEATURES,0);
		}
	}
	}
	// return true if a valid test has been found
	// test is invalid if only splits with an empty set A or B has been created
	return ret;
}

// generateTest: Generate test i.e. randomly selected dimension of input features
void CRTree::generateTest(int* test, unsigned int lengthVec)
{
	for(int i = 0; i < C_NUM_FEATURES; i++)
	{
		test[i] = cv::theRNG().uniform(0, lengthVec);
	}
}

// evaluateTest: Evaluate a random candidate split
void CRTree::evaluateTest(const cv::Mat &inData, cv::Mat &retFeatures, int* temp)
{
	retFeatures = cv::Mat::zeros(C_NUM_FEATURES, inData.cols, inData.type());

	for(int i = 0; i < inData.cols; i++)
	{
		for(int j = 0; j < C_NUM_FEATURES; j++)
		{
			retFeatures.at<float>(j,i) = inData.at<float>(temp[j],i);
		}
	}

}

// split: Partition data using a given split criteria - returns hashmap called partitionMap
//		  that indicates left and right partitions
void CRTree::split(cv::Mat &partitionMap, const cv::Mat &selFeatures, const cv::Mat &cThresholds)
{	
	// based on the thresholds and the values of each feature
	// evaluate split function to send each input sample to left or right subtree | using binary function h(v, theta) = 1 or 0

	for(int i = 0; i < selFeatures.cols; i++)
	{

		// perform check
		bool validationCheckL = true; 
		for(int j = 0; j < selFeatures.rows; j++)
		{
			validationCheckL &= (selFeatures.at<float>(j,i) < cThresholds.at<float>(j,0));
			//validationCheckR &= (selFeatures.at<float>(j,i) < cThresholds.at<float>(j,0));
			//std::cout << selFeatures.at<float>(j,i) << " > " << cThresholds.at<float>(j,0) << std::endl;
		}
		
		// save decision | 1 is saved if right set
		// saving nothing leaves the index to zeros meaning left node
		if(!validationCheckL)
		{
			//std::cout << "Right" << std::endl;
				partitionMap.at<float>(0,i) = 1;
		}
	}

	// nothing is returned by value - all done through reference to partitionMap
}

// measureInformationGain: Information gain energy calculator 
//          - takes partitionMap returned from split() and the corresponding target labels
double CRTree::measureInformationGain(const cv::Mat &inLabel, const cv::Mat &partitionMap)
{
	cv::Mat numOfElements;
	cv::reduce(partitionMap, numOfElements, 1 /*means reduced to single column*/ , CV_REDUCE_SUM);
	int numOfRight = numOfElements.at<float>(0,0);
	int numOfLeft = inLabel.cols - numOfRight;

	// IG = \log |\Sigm a(P)| - \sum_{i \in \{L, R\}} w_i \log |\Sigma_i (P_i)|
	// w_i = \frac{|P_i|}{|P|}
	double Wl = (double)numOfLeft/(double)inLabel.cols;
	double Wr = (double)numOfRight/(double)inLabel.cols;

	// Compute the covariance matrices for all data
	cv::Mat covP(0, 0, CV_32FC1);
	cv::Mat meanP(0, 0, CV_32FC1);
	cv::calcCovarMatrix(inLabel, covP, meanP, CV_COVAR_COLS | CV_COVAR_NORMAL | CV_COVAR_SCALE);    

	// prepare containers for left and right set
	// Left branch
	cv::Mat Pl(inLabel.rows, numOfLeft, CV_32FC1);

	// right branch
	cv::Mat Pr(inLabel.rows, numOfRight, CV_32FC1);

	int rIdx = 0, lIdx = 0;

	// split the data into two sets .// left and .// right
	for (unsigned i = 0; i < inLabel.cols; i++) {
		
		if(partitionMap.at<float>(0, i) == 0)
		{
			inLabel.col(i).copyTo(Pl.col(lIdx));
			lIdx++;
		}
		else
		{
			inLabel.col(i).copyTo(Pr.col(rIdx));
			rIdx++;
		}
	}
	// cov and mean for left branch
	cv::Mat covPl(0, 0, CV_32FC1);
	cv::Mat meanPl(0, 0, CV_32FC1);
	cv::calcCovarMatrix(Pl, covPl, meanPl, CV_COVAR_COLS | CV_COVAR_NORMAL | CV_COVAR_SCALE);        

	// cov and mean for right branch
	cv::Mat covPr(0, 0, CV_32FC1);
	cv::Mat meanPr(0, 0, CV_32FC1);
	cv::calcCovarMatrix(Pr, covPr, meanPr, CV_COVAR_COLS | CV_COVAR_NORMAL | CV_COVAR_SCALE);        

	double ig = log(cv::determinant(covP)) - Wr*log(cv::determinant(covPr)) - Wl*log(cv::determinant(covPl));

	if (isinf(ig)) {
		ig = 0; 
	}
	return ig;
}


// functions for displaying decision trees
void CRTree::displayDecisionTree(int idxImage)
{
	treeColor = cv::Scalar(rand()%156, rand()%156+100, rand()%56+200);

	int heightImage = (this->m_maxDepth)*C_NUM_VERTICAL_SPACE;
	int widthImage = (this->m_maxDepth)*C_NUM_HORIZONTAL_SPACE;

	this->displayTree = cv::Mat(heightImage+C_PADDING_DISPLAY/2, widthImage+C_PADDING_DISPLAY, CV_8UC3, cv::Scalar(190, 190, 190));
	
	// starting pt
	cv::Vec2i pt;

	pt[0] = (widthImage + C_PADDING_DISPLAY)/2;
	pt[1] = 20;
	if(this->m_treetable.cols>0)
		this->displayDecisionTree(pt, widthImage/2, 0, 0);

	cv::imshow("Tree", displayTree);
	cv::waitKey(10);

	//cv::Mat savedImage;
	//cv::flip(displayTree, savedImage, 0);
	char buffer[100];
	sprintf(buffer, "DisplayTrees\\tree_%05d.png", idxImage);
	cv::imwrite(buffer, displayTree);

}

void CRTree::displayDecisionTree(cv::Vec2i curPts, int realWidth, int cNode, int cDepth)
{
	stringStr.str("");
	if(cNode < m_treetable.cols-1)
	{
		if(this->m_treetable.at<float>(0, cNode) == -1) // if there is a node
		{
			
			// for left subtree repeat
			cv::Vec2i leftPos;
			leftPos[0] = curPts[0] - realWidth/2;
			leftPos[1] = curPts[1] + C_NUM_VERTICAL_SPACE;

			// draw the line
			int leftNode = m_treetable.at<float>(3, cNode);
			if(leftNode != 0)//63, 133, 205
			{
				// brown color --> cv::line(displayTree, curPts, leftPos, cv::Scalar(63, 133, 205), m_maxDepth-cDepth + 1);
				cv::line(displayTree, curPts, leftPos, treeColor, m_maxDepth-cDepth + 1);
			}
			displayDecisionTree(leftPos, realWidth/2, leftNode, cDepth+1);


			// for right subtree repeat
			cv::Vec2i rightPos;
			rightPos[0] = curPts[0] + realWidth/2;
			rightPos[1] = curPts[1] + C_NUM_VERTICAL_SPACE;

			int rightNode = m_treetable.at<float>(4, cNode);
			if(rightNode != 0)
			{
				// brown color --> cv::line(displayTree, curPts, rightPos, cv::Scalar(63, 133, 205), m_maxDepth-cDepth + 1);
				cv::line(displayTree, curPts, rightPos, treeColor, m_maxDepth-cDepth + 1);
			}
			displayDecisionTree(rightPos, realWidth/2, rightNode, cDepth+1);


		}
		else
		{
			// TODO:: leaf nodes -- maybe display some stats of each leaf


			// return back to collapse the recursive call
			return;
		}
	}
	else
		return;
}
