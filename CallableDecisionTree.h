#pragma once



#ifndef	CALLABLEDECISIONTREE_H
#define CALLABLEDECISIONTREE_H
#include"DecisionCoreFunction.h"


class CallableDecisionTree
{
private:
	const double getDataProbability = 0.5;

	std::shared_ptr<DecisionTreeNode>root;

	std::vector<Sample>trainSamples;
	
	std::vector<bool>remaindingFeature;
public:

	//获取数据的概率
	

	CallableDecisionTree(double _probability=0.5):root(std::make_shared<DecisionTreeNode>()),
		getDataProbability(_probability)
	{
		Utility::initRemaindingFeature(remaindingFeature);
	}
	~CallableDecisionTree() = default;


	void load();//加载数据
	void train();//训练
	//预测
	double predict(const Sample&);
};


#endif // !CALLABLEDECISIONTREE_H


