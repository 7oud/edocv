#ifndef __UTIL_H__
#define __UTIL_H__


struct LabeledSampleList
{
	vector<string> fileList;
	string label;
};

struct ANN_Train_Param
{
	int normWidth;			// 16
	int normHeight;			// 28

	int featureDim;			// 0

	int midLayerNeuronCnt;	// 30
	int maxIterCnt;			// 5000
	double param1;			// 0.05
	double param2;			// 0.00

};

size_t GetFilesList(string path, string ext, vector<string>& list);
int GetFilesList(string ext, string path, vector<LabeledSampleList>& list);

bool FindIdenticalImage(string path1, string path2);
int CheckIdenticalImage(string path);

int GetClassCount();

int MapLabel2Index(string label);
string MapIndex2Label(int idx);

Mat CalcImageFeature(Mat& imgSrc, Size normSize);

int makeTrainMatrix(string trainImgPath, vector<LabeledSampleList>& trainSamplesList, Size normSize, 
	Mat& matTrainData, Mat& matTrainResponse);

int makeSVMTrainMatrix(string trainImgPath, vector<LabeledSampleList>& trainSamplesList, Size normSize, 
	Mat& matTrainData, Mat& matTrainResponse);

#endif