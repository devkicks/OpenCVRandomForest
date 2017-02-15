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

#include "RForest.h"
// TODO:: Add verbrose mode for debugging
// constructor
CRForest::CRForest(unsigned int numOfTrees)
{
	m_numOfTrees = numOfTrees;
	decisionForest = new CRTree[m_numOfTrees];
}

// destructor
CRForest::~CRForest(void)
{
	delete [] decisionForest;
}

// Train: Training Random Forest using input data and regressors params
void CRForest::Train(const cv::Mat &inData, const cv::Mat &inLabel, const double& minVal, const double& maxVal )
{
	
	for(int i = 0; i  < m_numOfTrees; i++)
	{
		cv::vector<cv::Mat> dataRet;
		dataRet = baggingFunctionReplacement(inData, inLabel);

		std::cout << "Growing tree number:" << i+1 << std::endl;
		decisionForest[i].Train(dataRet[0], dataRet[1]);
	}

}

// Predict: Infer the target, given an input feature
cv::Mat CRForest::Predict(const cv::Mat &inData)
{
	CLeafNode result, temp;
	for(int i = 0; i < this->m_numOfTrees; i++)
	{
		temp = decisionForest[i].Predict(inData);
		//std::cout << "Mean : " << temp->m_mean << std::endl;
		if(i == 0)
		{
			result.m_cov = temp.m_cov.clone();
			result.m_mean = temp.m_mean.clone();
			//result->m_dataPoints = temp->m_dataPoints;
		}
		else
		{
		result.m_cov += temp.m_cov.clone();
		result.m_mean += temp.m_mean.clone();
		//result->m_dataPoints = temp->m_dataPoints;

		}
		
	}
	result.m_cov = result.m_cov/m_numOfTrees;
	result.m_mean = result.m_mean/m_numOfTrees;
	//std::cout << "Mean is: " <<  result->m_mean << " Covariance is: " << result->m_cov << std::endl;
	
	return result.m_mean.clone();
}

// TODO:: Improve the efficiency of without replacement function randSample
cv::vector<cv::Mat> CRForest::baggingFunction(const cv::Mat &inData, const cv::Mat &inLabel, float ratio)
{
	cv::vector<cv::Mat> retVector;

	// get the total number of samples and the num to be sampled based on the ratio
	int numOfData = inData.cols;
	int numOfSamples = int((double)numOfData * ratio);

	// sample random index without replacement from a defined total number of index
	cv::vector<int> indexOfSelected = randSample(numOfData, numOfSamples);

	// prepare containers for output data
	cv::Mat outData, outLabel;
	outData = cv::Mat::zeros(inData.rows, numOfSamples, inData.type());
	outLabel = cv::Mat::zeros(inLabel.rows, numOfSamples, inLabel.type());
	
	// copy the output data
	for(int i = 0; i < indexOfSelected.size(); i++)
	{
		inData.col(indexOfSelected[i]).copyTo(outData.col(i));
		inLabel.col(indexOfSelected[i]).copyTo(outLabel.col(i));
	}

	// push to a vector and return
	retVector.push_back(outData.clone());
	retVector.push_back(outLabel.clone());
	return retVector;
}

cv::vector<cv::Mat> CRForest::baggingFunctionReplacement(const cv::Mat &inData, const cv::Mat &inLabel, float ratio)
{
	cv::vector<cv::Mat> retVector;

	// get the total number of samples and the num to be sampled based on the ratio
	int numOfData = inData.cols;
	int numOfSamples = int((double)numOfData * ratio);

	// sample random index without replacement from a defined total number of index
	cv::vector<int> indexOfSelected = randSampleReplacement(numOfData, numOfSamples);

	// prepare containers for output data
	cv::Mat outData, outLabel;
	outData = cv::Mat::zeros(inData.rows, numOfSamples, inData.type());
	outLabel = cv::Mat::zeros(inLabel.rows, numOfSamples, inLabel.type());
	
	// copy the output data
	for(int i = 0; i < indexOfSelected.size(); i++)
	{
		inData.col(indexOfSelected[i]).copyTo(outData.col(i));
		inLabel.col(indexOfSelected[i]).copyTo(outLabel.col(i));
	}

	// push to a vector and return
	retVector.push_back(outData.clone());
	retVector.push_back(outLabel.clone());
	return retVector;
}
// randSample returns indices of randomly sampled samples
// from a total numOfData extracts numOfSamples
// without replacement 

//TODO: improve this implementation
cv::vector<int> CRForest::randSample(int numOfData, int numOfSamples)
{
	if(numOfSamples > numOfData)
	{
		std::cout << "Error: can not draw samples more than the data length without replacement" << std::endl;
		exit(EXIT_FAILURE);
	}

	cv::vector<int> idxRet;
	int temp;
	for(int i = 0; i < numOfSamples; i++)
	{
		int sizeIdx = idxRet.size();
		if(sizeIdx == 0)
			idxRet.push_back(rand() % numOfData);
		else
		{
			bool loop = true;
			while(loop)
			{
				temp = rand() % numOfData;
				for(int j = 0; j < sizeIdx; j++)
				{
					if(temp == idxRet[j])
					{
						// if the index is already sampled, then repeat random sampling
						loop = true;
						break;
					}
					else // if not found then prepare to exit loop and save the idx into vector
						loop = false;
				}
			}
			// save the index 
			idxRet.push_back(temp);
		}
	}
	return idxRet;
}

cv::vector<int> CRForest::randSampleReplacement(int numOfData, int numOfSamples)
{
	if(numOfSamples > numOfData)
	{
		std::cout << "Error: can not draw samples more than the data length without replacement" << std::endl;
		exit(EXIT_FAILURE);
	}

	cv::vector<int> idxRet;
	for(int i = 0; i < numOfSamples; i++)
	{
		idxRet.push_back(rand() % numOfData);
	}

	return idxRet;
}