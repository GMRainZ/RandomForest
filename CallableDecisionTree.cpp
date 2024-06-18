#include "CallableDecisionTree.h"


void CallableDecisionTree::load()
{
	//Utility::getTrainData(trainSamples,this->getDataProbability);
	Utility::getTrainData(trainSamples);

	//检查特征向量的值
	//Utility::checkFeature(trainSamples[0].feature);
	//Utility::checkFeature(trainSamples[500].feature);
	//exit(0);
}

void CallableDecisionTree::train()
{
	DecisionCoreFunction::generateTree(root,trainSamples,remaindingFeature);
}


double CallableDecisionTree::predict(const Sample& sample)
{
	return root->predict(sample);
}


