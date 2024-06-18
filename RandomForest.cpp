#include "RandomForest.h"

void RandomForest::train()
{
	for (int i = 0; i < numberOfTrees; ++i)
	{
		trees[i]->train();
	}
}

void RandomForest::predict()
{
	int i,predictNumber;
	std::vector<int>predictCount(10,0);


	for (const auto& sample : testSamples)
	{
		for (i = 0; i < numberOfTrees; ++i)
		{
			++predictCount[static_cast<int>(trees[i]->predict(sample))];
		}

		predictNumber = std::max_element(predictCount.cbegin(), predictCount.cend()) - predictCount.cbegin();
		printf("orginal number : %.0lf\nRandomForest predicts : %d\n", sample.label[0], predictNumber);

		predictCount.assign(10, 0);
	}

}
