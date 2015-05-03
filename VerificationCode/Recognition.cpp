#include "stdafx.h"



///////////////////////// feature calc //////////////////////////////////

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

