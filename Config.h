#pragma once


#ifndef CONFIG_H
#define CONFIG_H


namespace Config
{
	size_t dimensions = 1024;//����������ά��
	size_t labelDim = 1;

	const std::string tainSetPath = "dataSet/trainData";
	const std::string testSetPath = "dataSet/testData";

	const size_t trainDigitCount = 500, testDigitCount=20;//ѵ�����������ֵĸ���

	const size_t tRow = 32, tCol = 32;//�任֮��ͼ��Ĵ�С
	const cv::Size tSize(tRow, tCol);
	const cv::uint8_t mappingSize = 10;

	const int padding = 1;
}



#endif 


