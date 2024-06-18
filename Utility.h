#pragma once


#ifndef UTILITY_H
#define UTILITY_H

#include<vector>
#include<string>
#include<memory>
#include<opencv2/opencv.hpp>
#include<unordered_map>
#include<cmath>
#include<algorithm>

class Sample
{
public:
	std::vector<double>feature, label;

	Sample(std::vector<double>_feature, std::vector<double>_label)
		:feature(_feature), label(_label) {}

	Sample(const Sample&rhs):Sample(rhs.feature,rhs.label){}
	Sample(Sample&& rhs)noexcept :feature(rhs.feature), label(rhs.label){}
	Sample() = default;

	~Sample() = default;
};


namespace Utility
{


	void getTrainData(std::vector<Sample>&,double);
	void getTrainData(std::vector<Sample>&);
	void getTestData(std::vector<Sample>&);
	void getFiles(const std::string& path, std::vector<std::string>& files);
	void initRemaindingFeature(std::vector<bool>&);
	void initTreeNodeClassSize(std::vector<double>&);


	void Gray2Binary(cv::Mat&);

	void imageCut(cv::Mat&,int rowCount,int colCount);
	void boundaryImage(cv::Mat&);


	void checkFeature(const std::vector<double>&);
	void checkSample(const std::vector<Sample>&);
}



#endif













//Mat t = imread("dataSet/trainData/trainDigit0/pic_0.png",IMREAD_GRAYSCALE);
//Mat dst;

//if (t.empty())
//{
//	printf("fail to open t!\n");
//	return -1;
//}

//t.convertTo(dst, CV_32F);

////imshow("int", t);
//

//for (int i = 0; i < dst.rows; ++i)
//{
//	for (int j = 0; j < dst.cols; ++j)
//	{
//		cout << dst.at<cv::float16_t>(i, j) << ends;
//	}
//	cout << endl;
//}

////waitKey(0);