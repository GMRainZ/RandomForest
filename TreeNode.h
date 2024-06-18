#pragma once


#ifndef DECISIONTREENODE_H
#define DECISIONTREENODE_H
#include"Utility.h"

class DecisionTreeNode
{
private:

	bool isFinal;
	std::vector<double>nodeClass;
private:
	std::vector<std::shared_ptr<DecisionTreeNode>>children;


	size_t dim;
	double threshold;
public:
	
	double predict(const Sample&);
	void flag2Leaf(int _nodeClass);
	void flag2ProcessNode(int _dim, int _threshold);
	void addChild(std::shared_ptr<DecisionTreeNode>& child);

	DecisionTreeNode():isFinal(false)
	{
		Utility::initTreeNodeClassSize(nodeClass);
	}
	~DecisionTreeNode() = default;
};

#endif