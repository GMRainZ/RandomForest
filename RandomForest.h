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
		int i;//��ʼ��ɭ���е���
		for (i = 0; i < numberOfTrees; ++i)
		{
			trees[i] = std::make_shared<CallableDecisionTree>();
			trees[i]->load();
		}

		//��ʼ�����Լ�
		
		Utility::getTestData(testSamples);
	}


	void train();
	void predict();
};


#endif // !1



