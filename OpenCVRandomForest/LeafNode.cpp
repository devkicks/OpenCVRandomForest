
#include "LeafNode.h"

//TODO:: Add code for Kernel Density Estimation

CLeafNode::CLeafNode(void)
{
}


CLeafNode::~CLeafNode(void)
{
}


void CLeafNode::createLeafWithData(const cv::Mat &inLabel)
{
	cv::Mat P(inLabel.rows, inLabel.cols, CV_64F);
	for (int i = 0; i < inLabel.cols; i++) 
	{
		for(int j = 0; j < inLabel.rows; j++)
		{
			P.at<double>(j, i) = inLabel.at<float>(j, i);
		}
	}
	cv::calcCovarMatrix(P, m_cov, m_mean, CV_COVAR_COLS | CV_COVAR_NORMAL | CV_COVAR_SCALE);
}