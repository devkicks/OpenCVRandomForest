#pragma once
#include "common.h"

class CLeafNode
{
public:
	cv::Mat m_cov, m_mean;
public:
	CLeafNode(void);
	void createLeafWithData(const cv::Mat &inLabel);
	~CLeafNode(void);
};

