#include"Utility.h"
#include<io.h>
#include<random>
#include<chrono>
#include"Config.h"



void Utility::getTrainData(std::vector<Sample>& trainData,double probability)
{
	std::vector<std::string>digits;
	Utility::getFiles(Config::tainSetPath, digits);

	std::vector<double>feature(Config::dimensions), label(Config::labelDim);
	cv::Mat tmp;

	size_t cnt = 0, i, j;

	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
	
	std::uniform_real_distribution<double> urd(0.0, 1.0);

	for (const auto& digit : digits)
	{
		if (urd(generator) >= probability)continue;

		tmp = cv::imread(digit, cv::IMREAD_GRAYSCALE);
		//convolution(tmp);
		//cv::imshow("pic", tmp);
		//cv::waitKey(500);

		boundaryImage(tmp);


		resize(tmp, tmp, Config::tSize);

		//Gray2Binary(tmp);
		//cv::imshow("pic", tmp);
		//cv::waitKey(0);


		for (i = 0; i < Config::tRow; ++i)
		{
			for (j = 0; j < Config::tCol; ++j)
			{
				feature[i * Config::tCol + j] = tmp.at<uchar>(i, j);
			}
		}

		label[0] = cnt++ / Config::trainDigitCount;

		trainData.emplace_back(feature, label);
	}

	printf("%zu\n", cnt);
}

void Utility::getTrainData(std::vector<Sample>& trainData)
{
	std::vector<std::string>digits;
	Utility::getFiles(Config::tainSetPath, digits);

	std::vector<double>feature(Config::dimensions), label(Config::labelDim);
	cv::Mat tmp;

	size_t cnt = 0,i,j;
	for (const auto& digit : digits)
	{
		tmp = cv::imread(digit, cv::IMREAD_GRAYSCALE);
		//convolution(tmp);
		//cv::imshow("pic", tmp);
		//cv::waitKey(500);
		
		boundaryImage(tmp);


		resize(tmp, tmp, Config::tSize);

		//Gray2Binary(tmp);
		//cv::imshow("pic", tmp);
		//cv::waitKey(0);


		for (i = 0; i < Config::tRow; ++i)
		{
			for (j = 0; j < Config::tCol; ++j)
			{
				feature[i * Config::tCol + j] = tmp.at<uchar>(i, j);
			}
		}

		label[0] = cnt++ / Config::trainDigitCount;

		trainData.emplace_back(feature, label);
	}

	printf("%zu\n", cnt);
}

void Utility::getTestData(std::vector<Sample>& testData)
{
	std::vector<std::string>digits;
	Utility::getFiles(Config::testSetPath, digits);

	std::vector<double>feature(Config::dimensions), label(Config::labelDim);
	cv::Mat tmp;

	int cnt = 0, i, j;
	for (const auto& digit : digits)
	{
		tmp = cv::imread(digit, cv::IMREAD_GRAYSCALE);
		//convolution(tmp);

		boundaryImage(tmp);
		resize(tmp, tmp, Config::tSize);
		//Gray2Binary(tmp);

		for (i = 0; i < Config::tRow; ++i)
		{
			for (j = 0; j < Config::tCol; ++j)
			{
				feature[i * Config::tCol + j] = static_cast<double>(tmp.at<uchar>(i, j));
				//printf("%4.2lf", feature[i * Config::tCol + j]);
			}
		}
		label[0] = cnt++ / Config::testDigitCount;

		testData.emplace_back(feature, label);
	}


	printf("%zu\n", cnt);
}

void Utility::getFiles(const std::string& path, std::vector<std::string>& files)
{

	//文件句柄  
	long long hFile = 0;
	//文件信息，_finddata_t需要io.h头文件  
	struct _finddata_t fileinfo;
	std::string p;
	int i = 0;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			//如果是目录,迭代之  
			//如果不是,加入列表  
			if ((fileinfo.attrib & _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					getFiles(p.assign(path).append("\\").append(fileinfo.name), files);
			}
			else
			{
				files.push_back(p.assign(path).append("\\").append(fileinfo.name));
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}


void Utility::initRemaindingFeature(std::vector<bool>& remaindingFeature)
{
	remaindingFeature.assign(Config::dimensions, true);
}
void Utility::initTreeNodeClassSize(std::vector<double>&nodeClass)
{
	nodeClass.assign(Config::labelDim,.0);
}

void Utility::imageCut(cv::Mat&inputImage, int rowCount, int colCount)
{
	std::string savePath;

	cv::Mat t;
	cv::Range rowRange, colRange;
	const cv::Size every(200, 200);
	const int rowSize = inputImage.rows / rowCount / 10, colSize = inputImage.cols / colCount;
	int k, i, j;
	for (k = 0; k < 10; ++k)
	{
		for (i = 0; i < rowCount; ++i)
		{
			for (j = 0; j < colCount; ++j)
			{
				rowRange = cv::Range((k + i) * rowSize, (k + i) * rowSize + rowSize);
				colRange = cv::Range(j * colSize, j * colSize + colSize);

				t = inputImage(rowRange, colRange);

				savePath = "dataSet/testData/testDigit" + std::to_string(k) + "/pic_" + std::to_string(i * colSize + j)+".png";


				if (cv::imwrite(savePath, t))
				{
					printf("successfully save\n");
				}
			}
		}
	}


}

void Utility::boundaryImage(cv::Mat&inputImage)
{
	int i, j,leftBod,rightBod,upBod,downBod;
	
	const int rowPixel = inputImage.rows, colPixel = inputImage.cols;
	

	//cv::cvtColor(inputImage, inputImage, cv::COLOR_BGR2GRAY);
	for (i = 0; i < rowPixel; ++i)
	{
		for (j = 0; j < colPixel; ++j)
		{
			if (inputImage.at<uchar>(i, j))break;
		}
		if (j < colPixel)break;
	}
	upBod = i;
	//printf("upBod = %d\n", upBod);

	for (i = rowPixel-1; i >=0; --i)
	{
		for (j = 0; j < colPixel; ++j)
		{
			if (inputImage.at<uchar>(i, j))break;
		}
		if (j < colPixel)break;
	}
	downBod = i;
	//printf("downBod = %d\n", downBod);


	for (j = 0; j < colPixel; ++j)
	{
		for (i = 0; i < rowPixel; ++i)
		{
			if (inputImage.at<uchar>(i, j))break;
		}
		if (i < rowPixel)break;
	}
	leftBod = j;
	//printf("leftBod = %d\n", leftBod);

	for (j = colPixel-1; j >= 0; --j)
	{
		for (i = 0; i < rowPixel; ++i)
		{
			if (inputImage.at<uchar>(i, j))break;
		}
		if (i < rowPixel)break;
	}
	rightBod = j;
	//printf("rightBod = %d\n", rightBod);


	inputImage = inputImage(cv::Range(upBod,downBod), cv::Range(leftBod,rightBod));

}



void Utility::Gray2Binary(cv::Mat&src)
{
	const int srow = src.rows, scol = src.cols;
	int i, j;
	for (i = 0; i < srow; ++i)
	{
		for (j = 0; j < scol; ++j)
		{
			if (src.at<uchar>(i, j) >= 127)src.at<uchar>(i, j) = 255;
			else src.at<uchar>(i, j) = 0;
		}
	}
}

void Utility::checkFeature(const std::vector<double>&feature)
{
	int i, j;
	for (i = 0; i < Config::tRow; ++i)
	{
		for (j = 0; j < Config::tCol; ++j)
		{
			printf("%-4.0lf ", feature[i * Config::tCol + j]);
		}
		printf("\n");
	}

}

void Utility::checkSample(const std::vector<Sample>& samples)
{
	int cnt = 0;
	for (const auto& sample : samples)
	{
		if (cnt % Config::trainDigitCount == 0)
			checkFeature(sample.feature);
	}
}