#include"TreeNode.h"

double DecisionTreeNode::predict(const Sample&sample)
{
	
	if (this->isFinal)
	{
		//sample.label[0] = this->nodeClass[0];
		return this->nodeClass[0];
	}
	
	return sample.feature[this->dim] >= this->threshold ? this->children[1]->predict(sample) :
		this->children[0]->predict(sample);
}

void DecisionTreeNode::flag2Leaf(int _nodeClass)
{
	//dim = _dim;
	//threshold = _thershold;
	isFinal = true;
	nodeClass[0] = _nodeClass;

	children.resize(0);
}

void DecisionTreeNode::flag2ProcessNode(int _dim, int _threshold)
{
	dim = _dim;
	threshold = _threshold;
}

void DecisionTreeNode::addChild(std::shared_ptr<DecisionTreeNode>& child)
{
	children.emplace_back(child);
}
