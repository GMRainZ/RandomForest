#include"DecisionCoreFunction.h"


double DecisionCoreFunction::crossEntropy(const std::unordered_map<double,int>&classCounts,int totalCount)
{
	//std::unordered_map<double, int>classCounts;

	//size_t totalCount = 0;
	//for (const auto& sample : samples)
	//{
	//	++classCounts[sample.label[0]];
	//	++totalCount;
	//}

	if (!totalCount)return .0;

	double ans = 0, t;

	for (const auto& classCount : classCounts)
	{
		/********************************************************/
		//未进行强制转换造成了bug
		/********************************************************/
		t = 1.*classCount.second / totalCount;
		ans += -t * log2(t);

		//printf("t=%5lf\tans=%5lf\n", t, ans);

	}


	return ans;
}

void DecisionCoreFunction::convolution(cv::Mat&src, ConvolutionKernel kernel)
{
	if (kernel == ConvolutionKernel::Laplace)cv::Laplacian(src, src, -1, 3);
	else if (kernel == ConvolutionKernel::Sobel)cv::Sobel(src, src, -1, 1, 0, 3);
}

double DecisionCoreFunction::crossEntropy(const std::vector<Sample>& samples)
{
	std::unordered_map<double, int>classCounts;

	size_t totalCount = 0;
	for (const auto& sample : samples)
	{
		++classCounts[sample.label[0]];
		++totalCount;
	}

	if (!totalCount)return .0;

	double ans = 0, t;

	for (const auto& classCount : classCounts)
	{
		t = classCount.second / totalCount;
		ans += -t * log2(t);
	}


	return ans;
}

double DecisionCoreFunction::calculateGain(const std::vector<Sample>& samples, const size_t dim)
{
	const int nSample = samples.size();


	//printf("nSample=%d\n", nSample);

	//false==minus   ,true==plus
	std::vector < bool >eignFlag(nSample,false);


	int i,minusCount=0,plusCount=0;

	for (i = 0; i < nSample; ++i)
	{
		if (samples[i].feature[dim]>=127.)eignFlag[i] = true, ++plusCount;
		else ++minusCount;
	}

	if (!plusCount || !minusCount)
	{
		//printf("plusCount=%d,minusCount=%d\n",plusCount,minusCount);

		return -1;
	}

	std::unordered_map<double, int>minusClassCounts, plusClassCounts;
	for (i = 0; i < nSample; ++i)
	{
		if (eignFlag[i])++plusClassCounts[samples[i].label[0]];
		else ++minusClassCounts[samples[i].label[0]];
	}
	
	return (1. * plusCount / nSample) * crossEntropy(plusClassCounts, plusCount) +
		(1. * minusCount / nSample) * crossEntropy(minusClassCounts, minusCount);

}

std::pair<double,double> DecisionCoreFunction::calculateGainForContinuousValue(const std::vector<Sample>&samples, const size_t dim)
{
	const double DCrossEntropy = crossEntropy(samples);

	std::vector<double>fDim;
	for (const auto& sample : samples)
		fDim.emplace_back(sample.feature[dim]);


	sort(fDim.begin(), fDim.end());

	const int n = fDim.size();
	

	int i;
	double sill, splitEnt, maxSplitEnt=-1., maxSplitEntIndex;
	
	std::vector<Sample>DMinus, DPlus;
	double crossEntropyMinus, crossEntropyPlus;
	int countMinus, countPlus;
	for (i = 0; i < n - 1; ++i)
	{
		sill = (fDim[i] + fDim[i + 1]) / 2;

		countMinus=countPlus=0;
		for (const auto& sample : samples)
		{
			if (sample.feature[dim] <= sill)
				DMinus.emplace_back(sample),++countMinus;
			else
				DPlus.emplace_back(sample),++countPlus;
		}

		crossEntropyMinus=crossEntropy(DMinus);
		crossEntropyPlus=crossEntropy(DPlus);

		splitEnt=DCrossEntropy - (1. * countMinus / n * crossEntropyMinus + 1. * countPlus / n * crossEntropyPlus);

		if (splitEnt > maxSplitEnt)
		{
			maxSplitEnt = splitEnt;
			maxSplitEntIndex = sill;
		}
	}
	
	//maxSplitEnt==该维度上的最大信息熵减
	//maxSplitEnt==在该维度上最佳的分割点
	return { maxSplitEnt,maxSplitEntIndex };
}

void DecisionCoreFunction::generateTree(std::shared_ptr<DecisionTreeNode>&curNode, //当前结点
	std::vector<Sample>&samples/*当前结点处所剩余的结点*/, 
	std::vector<bool> remaindingFeature/*当前结点可用的特征*/, decisionRule dr)
{
	//记录各类别的个数
	std::unordered_map<double, int>classCount;
	double maxCountClass=-1, maxCount=-1;
	
	for (const auto& sample : samples)
	{
		//if (classCount.size() > 1)break;
		++classCount[sample.label[0]];
		if (maxCount >= classCount[sample.label[0]]||
			sample.label[0] == maxCountClass)continue;
		//if (sample.label[0] == maxCountClass)continue;
		
		maxCount = classCount[sample.label[0]];
		maxCountClass = sample.label[0];
		
	}
	//样本中类别都相同
	if (classCount.size() <= 1)
	{
		curNode->flag2Leaf(classCount.begin()->first);

		//printf("类别都相同，标签为%.0lf\n", classCount.begin()->first);

		return;
	}

	//maxSplitEnt==该维度上的最大信息熵减
	//maxSplitDim==在该维度上最佳的分割点
	double maxSplitEnt = -1., splitEnt=0.;

	int splitDim=-1;
	const int dimensions = remaindingFeature.size();
	int i; bool hasRemaind=false,hasDiff=false;

	if (dr == decisionRule::Gain)
	{
		for (i = 0; i < dimensions; ++i)
		{
			//该维度已经使用过，继续搜索
			if (!remaindingFeature[i])continue;

			//表示至少还一一个剩余，进入计算
			hasRemaind = true;// printf("hasRemaind已经被修改\n");
			splitEnt = calculateGain(samples, i);
			//printf("splitEnt = %5lf\n\n", splitEnt);
			//当前计算后发现所有结点的特征都是一样的
			if (splitEnt == -1)continue;

			//特征不一样则hasdiff置位
			hasDiff = true; //printf("hasDiff已经被修改\n");


			if (splitEnt > maxSplitEnt)
			{
				maxSplitEnt = splitEnt;
				splitDim = i;
				//printf("splitDim = %d\n", splitDim);
			}
		}
	}
	else if (dr == decisionRule::Gain_ratio)
	{
		printf("running\n");
	}
	else if (dr == decisionRule::Gini_index)
	{
		printf("running\n");
	}

	//已经没有特征可选了 或者  在当前剩余特征集合上所有样本特征相同//直接标记为类别最多的
	if (!hasRemaind || !hasDiff)
	{
		curNode->flag2Leaf(maxCountClass);
		//printf("无特征可选，标签为%.0lf\n",maxCountClass);
		return;
	}

	//不可能标记成叶子结点了，继续往下递归
	if (splitDim == -1)
	{
		printf("this is a bug\n");
		return ;
	}
	
	
	//printf("本次使用的分割维度为%d\n", splitDim);
	remaindingFeature[splitDim] = false;

	std::vector<Sample>DMinus, DPlus;
	for (auto& sample : samples)
	{
		if (sample.feature[splitDim]>=127)DPlus.emplace_back(std::move(sample));
		else DMinus.emplace_back(std::move(sample));
	}

	//给当前结点创建孩子节点
	std::shared_ptr<DecisionTreeNode>lchild=std::make_shared<DecisionTreeNode>(), 
		rchild = std::make_shared<DecisionTreeNode>();

	curNode->flag2ProcessNode(splitDim, 127);
	//printf("我不是叶子节点\n");


	//递归产生树
	generateTree(lchild, DMinus, remaindingFeature);
	generateTree(rchild, DPlus, remaindingFeature);

	
	//添加到记录中
	curNode->addChild(lchild);
	curNode->addChild(rchild);
	//左孩子为MinusClass , 右孩子为PlusClass


}



