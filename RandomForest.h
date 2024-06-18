#pragma once





#ifndef RANDOMFOREST_H
#define RANDONFOREST_H

#include"CallableDecisionTree.h"

class RandomForest
{
private:
	static const int numberOfTrees=10;

	std::shared_ptr<CallableDecisionTree>trees[numberOfTrees];
	std::vector<Sample>testSamples;
public:
	RandomForest()
	{
		int i;//初始化森林中的树
		for (i = 0; i < numberOfTrees; ++i)
		{
			trees[i] = std::make_shared<CallableDecisionTree>();
			trees[i]->load();
		}

		//初始化测试集
		
		Utility::getTestData(testSamples);
	}


	void train();
	void predict();
};


#endif // !1



