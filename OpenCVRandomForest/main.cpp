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

Usage:

The trees are encapsulated into Forest.
The models assume that the data is arranged in columns i.e. each column has a new sample in data

Initializing a Forest structure
CRForest myForest(numOfTrees)

Training:
myForest.Train(trainFeatures, trainTarget)

Prediction:
cv::Mat predOut = myForest.Predict(testFeature);

Seperate matlab files are provided to analyse the output in Matlab

*/

#include "RForest.h"
#include <fstream>

void writeMatlabFile(std::string filename, const cv::Mat &inData);
cv::Mat readMatlabFile(std::string filename);
cv::Mat createTestValues(const cv::Mat &inData);
cv::Mat calculateAverageError( cv::Mat predLabels, cv::Mat gtLabels);
void loadRoadData(cv::vector<cv::Mat> &trainSet, cv::vector<cv::Mat> &testSet);
void loadAirfoilData(cv::vector<cv::Mat> &trainSet, cv::vector<cv::Mat> &testSet);
void loadBreastCancerData(cv::vector<cv::Mat> &trainSet, cv::vector<cv::Mat> &testSet);

int main()
{
	cv::vector<cv::Mat> trainSet, testSet;

	// Try other datasets using
	// loadRoadData(trainSet, testSet); // Too much data - takes a while to train
	 loadAirfoilData(trainSet, testSet);
	//loadBreastCancerData(trainSet, testSet); // Insufficient data for fully training
	 int numOfTrees = 100;
	CRForest myForest(numOfTrees);

	// train the random forest
	myForest.Train(trainSet[0], trainSet[1]);

	// display tree structure
	for(int i = 0; i < numOfTrees; i++)
	{
		myForest.displayTree(i);
	}

	int numOfTestSamples = testSet[0].cols;
	cv::Mat labelPred = cv::Mat::zeros(testSet[1].rows, numOfTestSamples, testSet[1].type());
	cv::Mat testLabel = cv::Mat::zeros(testSet[1].rows, numOfTestSamples, testSet[1].type());
	for(int i= 0; i < numOfTestSamples; i++)
	{
		std::cout << (float)i/(float)numOfTestSamples << std::endl;
		myForest.Predict(testSet[0].col(i)).clone().copyTo(labelPred.col(i));
		testSet[1].col(i).copyTo(	testLabel.col(i) );
	}

	std::cout << "Size GT Labels: " << testSet[1].cols << ", " << testSet[1].rows << std::endl;
	std::cout << "Size Pred Labels: " << labelPred.cols << ", " << labelPred.rows << std::endl;
	std::cout << "Mean error: " << calculateAverageError(labelPred, testLabel) << std::endl;

	// Write dat file to analayse output in Matlab
	writeMatlabFile("Datasets\\results\\Pred.dat", labelPred);
	writeMatlabFile("Datasets\\results\\GT.dat", testSet[1]);

	return 1;

}

cv::Mat calculateAverageError( cv::Mat predLabels, cv::Mat gtLabels)
{
	double retVal1 = 0, retVal2 = 0 ;

	if(gtLabels.cols != predLabels.cols)
	{
		std::cout << "Error calculate average: size of both mat should be same" <<std::endl;
		exit(0);
	}

	cv::Mat retAvgError = cv::Mat::zeros(gtLabels.rows, 1, CV_32FC1);

	cv::reduce(cv::abs(gtLabels-predLabels), retAvgError, 1 /*means reduced to single column*/ , CV_REDUCE_AVG);

	return retAvgError;
}

void writeMatlabFile(std::string filename, const cv::Mat& inData)
{
	std::fstream file;
	//cv::Mat retMat;
	file.open(filename.c_str(), std::ios::out | std::ios::binary);
	if(!file.is_open())
	{
		std::cout << "Error opening matlab binary file" << std::endl;
		std::cout << filename << std::endl;
	}

	// read the size of the Mat (including the number of channels)
	double colsMat = (double)inData.cols, rowsMat = (double)inData.rows, channelsMat = (double)inData.channels();
	
	file.write((char*)&rowsMat, sizeof(rowsMat));
	file.write((char*)&colsMat, sizeof(colsMat));

	for(int i = 0; i < inData.cols; i++)
	{
		for(int j = 0; j < inData.rows; j++)
		{
			double outElement = (double)inData.at<float>(j,i);
			file.write((char*)&outElement, sizeof(double));
		}
	}
	file.close();
}

cv::Mat readMatlabFile(std::string filename)
{
	std::fstream file;
	cv::Mat retMat;
	file.open(filename.c_str(), std::ios::in | std::ios::binary);
	if(!file.is_open())
	{
		std::cout << "Error opening matlab binary file" << std::endl;
		std::cout << filename << std::endl;
		//isProbLoaded = false;
		//return false;		
	}

	// read the size of the Mat (including the number of channels)
	double colsMat, rowsMat, channelsMat;

	file.read((char*)&rowsMat, sizeof(rowsMat));
	file.read((char*)&colsMat, sizeof(colsMat));
	//file.read((char*)&channelsMat, sizeof(channelsMat));


	retMat = cv::Mat::zeros((int)rowsMat, (int)colsMat, CV_32FC1);

	//for(int k = 0; k < retMat.channels(); k++)
	//{
	for(int i = 0; i < retMat.cols; i++)
	{

		for(int j = 0; j < retMat.rows; j++)
		{

			double buff;
			file.read((char*)&buff, sizeof(buff));
			//file.read((char*)&m_ratioProb.at<double>(j, i), sizeof(m_ratioProb.at<double>(j, i)));
			retMat.at<float>(j,i) = (float)buff;
		}
	}
	//}
	//isProbLoaded = true;
	file.close();
	return retMat;
}

void loadRoadData(cv::vector<cv::Mat> &trainSet, cv::vector<cv::Mat> &testSet)
{
	cv::Mat buffer;
	std::string folderName = "Datasets\\3D_RoadNetwork\\Singlefold\\";
	buffer = readMatlabFile(folderName + std::string("RoadNetwork_TrainFeatures.dat"));
	trainSet.push_back(buffer.clone());

	buffer.release();

	buffer = readMatlabFile(folderName + std::string("RoadNetwork_TestFeatures.dat"));
	testSet.push_back(buffer.clone());

	buffer.release();

	buffer = readMatlabFile(folderName + std::string("RoadNetwork_TrainTarget.dat"));
	trainSet.push_back(buffer.clone());

	buffer.release();

	buffer = readMatlabFile(folderName + std::string("RoadNetwork_TestTarget.dat"));
	testSet.push_back(buffer.clone());

	buffer.release();

	//all done
}

void loadAirfoilData(cv::vector<cv::Mat> &trainSet, cv::vector<cv::Mat> &testSet)
{
	cv::Mat buffer;
	std::string folderName = "Datasets\\AirFoil\\Singlefold\\";
	buffer = readMatlabFile(folderName + std::string("Airfoil_TrainFeatures.dat"));
	trainSet.push_back(buffer.clone());

	buffer.release();

	buffer = readMatlabFile(folderName + std::string("AirFoil_TestFeatures.dat"));
	testSet.push_back(buffer.clone());

	buffer.release();

	buffer = readMatlabFile(folderName + std::string("AirFoil_TrainTarget.dat"));
	trainSet.push_back(buffer.clone());

	buffer.release();

	buffer = readMatlabFile(folderName + std::string("AirFoil_TestTarget.dat"));
	testSet.push_back(buffer.clone());

	buffer.release();

	//all done
}


void loadBreastCancerData(cv::vector<cv::Mat> &trainSet, cv::vector<cv::Mat> &testSet)
{
	cv::Mat buffer;
	std::string folderName = "Datasets\\BreastCancer\\Singlefold\\";
	buffer = readMatlabFile(folderName + std::string("BreastCancer_TrainFeatures.dat"));
	trainSet.push_back(buffer.clone());

	buffer.release();

	buffer = readMatlabFile(folderName + std::string("BreastCancer_TestFeatures.dat"));
	testSet.push_back(buffer.clone());

	buffer.release();

	buffer = readMatlabFile(folderName + std::string("BreastCancer_TrainTargetRegression.dat"));
	trainSet.push_back(buffer.clone());

	buffer.release();

	buffer = readMatlabFile(folderName + std::string("BreastCancer_TestTargetRegression.dat"));
	testSet.push_back(buffer.clone());

	buffer.release();

	//all done
}
