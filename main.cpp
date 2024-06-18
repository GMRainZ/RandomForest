#include"RandomForest.h"

using namespace std;
using namespace cv;



int main()
{
	
	RandomForest rf;

	rf.train();
	rf.predict();

	return 0;
}