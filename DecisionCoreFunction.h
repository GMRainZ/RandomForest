#pragma once


#ifndef DECISIONCOREFUNTION_H
#define DECISIONCOREFUNTION_H

#include"TreeNode.h"

enum  class decisionRule :std::uint8_t
{
	Gain,
	Gain_ratio,
	Gini_index
};

enum class ConvolutionKernel :std::uint8_t 
{ 
	Sobel, 
	Laplace 
};

namespace DecisionCoreFunction
{
	
	void convolution(cv::Mat&, ConvolutionKernel kernel = ConvolutionKernel::Laplace);

	double crossEntropy(const std::vector<Sample>&);
	double crossEntropy(const std::unordered_map<double, int>& classCount,int );//¼ÆËã½»²æìØ
	double calculateGain(const std::vector<Sample>&, const size_t);//
	double calculateGain_ratio(const std::vector<Sample>&, const size_t);
	double calculateGini(const std::vector<Sample>&, const size_t);
	double calculateGini_index(const std::vector<Sample>&, const size_t);
	double calculateIV(const std::vector<Sample>&, const size_t);


	void generateTree(std::shared_ptr<DecisionTreeNode>&,std::vector<Sample>&, std::vector<bool>, decisionRule dr = decisionRule::Gain);
	

	std::pair<double, double>calculateGainForContinuousValue(const std::vector<Sample>& samples, const size_t dim);
}


#endif // !DECISIONCOREFUNTION_H
