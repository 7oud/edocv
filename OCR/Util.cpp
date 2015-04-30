#include "stdafx.h"

#include "Util.h"


//////////////////////////////////////////////////////////////////////////


size_t GetFilesList(string path, string ext, vector<string>& list)
{
	string filespec = path + "*." + ext;

	struct _finddata_t fileinfo;
	intptr_t handle;

	if ((handle = _findfirst(filespec.c_str(), &fileinfo)) != -1L)
	{
		do 
		{
			if (fileinfo.name[0] == '.')
				continue;

			if (_A_SUBDIR & fileinfo.attrib)
				continue;

			int pos = string(fileinfo.name).find_first_of("_");
			char level = fileinfo.name[pos+1];

			list.push_back(string(fileinfo.name));

		} while(_findnext(handle, &fileinfo) == 0);

		_findclose(handle);
	}

	return list.size();
}


int GetFilesList(string path, string ext, vector<LabeledSampleList>& list)
{
	string filespec = path + "*.*";

	struct _finddata_t fileinfo;
	intptr_t handle;

	if ((handle = _findfirst(filespec.c_str(), &fileinfo)) != -1L)
	{
		do 
		{
			if (fileinfo.name[0] == '.')
				continue;

			if (_A_SUBDIR & fileinfo.attrib)
			{
				LabeledSampleList vec;
				vec.label = fileinfo.name;

				string subPath = path + fileinfo.name + "/";
				GetFilesList(subPath, ext, vec.fileList);

				list.push_back(vec);
			}

		} while(_findnext(handle, &fileinfo) == 0);

		_findclose(handle);
	}

	return list.size();
}


//////////////////////////////////////////////////////////////////////////


bool IsIdenticalImage(Mat& img1, Mat& img2)
{
	if (img1.rows != img2.rows || 
		img1.cols != img2.cols ||
		img1.channels() != img2.channels())
	{
		return false;
	}

	for (int i = 0; i < img1.rows; i++)
	{
		for (int j = 0; j < img1.cols; j++)
		{
			uchar* elem1 = &((uchar*)(img1.data + i * img1.step))[j*3];
			uchar* elem2 = &((uchar*)(img2.data + i * img2.step))[j*3];

			if (*elem1 != *elem2)
				return false;
		}
	}

	return true;
}


bool FindIdenticalImage(string path1, string path2)
{
	vector<string> fileList1;
	GetFilesList(path1, "BMP", fileList1);

	vector<string> fileList2;
	GetFilesList(path2, "BMP", fileList2);

	for (int i = 0; i < fileList1.size(); i++)
	{
		for (int j = 0; j < fileList2.size(); j++)
		{
			string fileName1 = path1 + fileList1[i];
			string fileName2 = path2 + fileList2[j];

			Mat img1 = imread(fileName1, 4);
			Mat img2 = imread(fileName2, 4);

			if (IsIdenticalImage(img1, img2))
			{
				cout << fileList1[i] << "\t" << fileList2[j] << endl;
				return true;
			}
		}
	}

	return false;
}


int CheckIdenticalImage(string path)
{
	vector<string> fileList;
	GetFilesList(path, "BMP", fileList);

	for (int i = 0; i < fileList.size(); i++)
	{
		for (int j = 0; j < fileList.size(); j++)
		{
			if (i != j)
			{
				string fileName1 = path + fileList[i];
				string fileName2 = path + fileList[j];

				Mat img1 = imread(fileName1, 4);
				if (img1.data == NULL)
					break;
				Mat img2 = imread(fileName2, 4);
				if (img2.data == NULL)
					continue;

				if (IsIdenticalImage(img1, img2))
				{
					cout << fileList[i] << " " << fileList[j] << endl;
					DeleteFile(fileName2.c_str());
					cout << "delete " << fileList[j] << endl;
				}
			}
		}
	}


	return 0;
}


///////////////////////// Label MAPPING //////////////////////////////////

string mlp_EnNum_label[] = {
	"2", "3", "4", "5", "7", "8", "A", "B", "C", "D", 
	"dd", "E", "ee", "F", "ff", "G", "H", "hh", "J", "K", 
	"L", "M", "N", "nn", "P", "Q", "R", "T", "tt", "V", "W", "X"};


int MapLabel2Index(string label)
{
	int idx = -1;

	if		(label == "2")		idx = 0;
	else if (label == "3")		idx = 1;
	else if (label == "4")		idx = 2;
	else if (label == "5")		idx = 3;
	else if (label == "7")		idx = 4;
	else if (label == "8")		idx = 5;
	else if (label == "A")		idx = 6;
	else if (label == "B")		idx = 7;
	else if (label == "C")		idx = 8;
	else if (label == "D")		idx = 9;
	else if (label == "dd")		idx = 10;
	else if (label == "E")		idx = 11;
	else if (label == "ee")		idx = 12;
	else if (label == "F")		idx = 13;
	else if (label == "ff")		idx = 14;
	else if (label == "G")		idx = 15;
	else if (label == "H")		idx = 16;
	else if (label == "hh")		idx = 17;
	else if (label == "J")		idx = 18;
	else if (label == "K")		idx = 19;
	else if (label == "L")		idx = 20;
	else if (label == "M")		idx = 21;
	else if (label == "N")		idx = 22;
	else if (label == "nn")		idx = 23;
	else if (label == "P")		idx = 24;
	else if (label == "Q")		idx = 25;
	else if (label == "R")		idx = 26;
	else if (label == "T")		idx = 27;
	else if (label == "tt")		idx = 28;
	else if (label == "V")		idx = 29;
	else if (label == "W")		idx = 30;
	else if (label == "X")		idx = 31;

	return idx;
}


string MapIndex2Label(int idx)
{
	return mlp_EnNum_label[idx];
}


int GetClassCount()
{
	return sizeof(mlp_EnNum_label) / sizeof(mlp_EnNum_label[0]);;
}


///////////////////////// Calculate Feature //////////////////////////////////


void deskew(Mat& img, Size SZ)
{
	Moments m = moments(img);

	if (m.mu02 < 1e-2)
		return;

	double skew = m.mu11 / m.mu02;

	Mat warp(2, 3, CV_32FC1);

	warp.at<float>(0, 0) = 1;
	warp.at<float>(0, 1) = skew;
	warp.at<float>(0, 2) = -0.5*SZ.width*skew;
	warp.at<float>(1, 0) = 0;
	warp.at<float>(1, 1) = 1;
	warp.at<float>(1, 2) = 0;

	warpAffine(img, img, warp, SZ, WARP_INVERSE_MAP | INTER_LINEAR);
}


Mat CalcImageFeature(Mat& imgSrc, Size normSize)
{
	Mat imgGray(imgSrc.rows, imgSrc.cols, CV_8UC1);

	if (imgSrc.channels() == 1)
		imgSrc.copyTo(imgGray);
	else
		cvtColor(imgSrc, imgGray, CV_BGR2GRAY);

	Mat imgNorm;
	resize(imgGray, imgNorm, normSize);

	deskew(imgNorm, normSize);

	Mat imgRowVec;
	imgNorm.convertTo(imgRowVec, CV_32FC1);

	return imgRowVec.reshape(0, 1);
}


int makeTrainMatrix(string trainImgPath, vector<LabeledSampleList>& trainSamplesList, Size normSize, 
	Mat& matTrainData, Mat& matTrainResponse)
{
	int clsCnt = trainSamplesList.size();

	size_t smplCnt = 0;
	for (size_t i = 0; i < trainSamplesList.size(); i++)
		smplCnt += trainSamplesList[i].fileList.size();

	cout << "Train Sample count = " << smplCnt << endl;
	cout << "Class count = " << clsCnt << endl;

	matTrainData.create(smplCnt, normSize.width * normSize.height, CV_32FC1);
	//matTrainResponse = Mat::zeros(smplCnt, 1, CV_32FC1);
	matTrainResponse = Mat::zeros(smplCnt, clsCnt, CV_32FC1);

	int rowIdx = 0;
	for (size_t i = 0; i < trainSamplesList.size(); i++)
	{
		for (size_t j = 0; j < trainSamplesList[i].fileList.size(); j++)
		{
			string sampleName = trainImgPath + "\\" + trainSamplesList[i].label + "\\" + trainSamplesList[i].fileList[j];
			
			Mat img = imread(sampleName, 4);
			if (img.data == NULL)
				continue;

			Mat mat = CalcImageFeature(img, normSize);
			mat.copyTo(matTrainData.row(rowIdx));

			int loc = MapLabel2Index(trainSamplesList[i].label);
			//matTrainResponse.at<float>(rowIdx, 0) = loc;
			matTrainResponse.at<float>(rowIdx, loc) = 1.0f;

			rowIdx++;
		}

		cout << trainSamplesList[i].label << " " << trainSamplesList[i].fileList.size() << endl;
	}

	return 0;
}


int makeSVMTrainMatrix(string trainImgPath, vector<LabeledSampleList>& trainSamplesList, Size normSize, 
	Mat& matTrainData, Mat& matTrainResponse)
{
	int clsCnt = trainSamplesList.size();

	size_t smplCnt = 0;
	for (size_t i = 0; i < trainSamplesList.size(); i++)
		smplCnt += trainSamplesList[i].fileList.size();

	cout << "Train Sample count = " << smplCnt << endl;
	cout << "Class count = " << clsCnt << endl;

	matTrainData.create(smplCnt, normSize.width * normSize.height, CV_32FC1);
	matTrainResponse = Mat::zeros(smplCnt, 1, CV_32SC1);

	int rowIdx = 0;
	for (size_t i = 0; i < trainSamplesList.size(); i++)
	{
		for (size_t j = 0; j < trainSamplesList[i].fileList.size(); j++)
		{
			string sampleName = trainImgPath + "\\" + trainSamplesList[i].label + "\\"  + trainSamplesList[i].fileList[j];

			Mat img = imread(sampleName, 4);
			if (img.data == NULL)
				continue;

			Mat mat = CalcImageFeature(img, normSize);
			mat.copyTo(matTrainData.row(rowIdx));

			int loc = MapLabel2Index(trainSamplesList[i].label);
			matTrainResponse.at<int>(rowIdx, 0) = loc;

			rowIdx++;
		}

		cout << trainSamplesList[i].label << " " << trainSamplesList[i].fileList.size() << endl;
	}

	return 0;
}

