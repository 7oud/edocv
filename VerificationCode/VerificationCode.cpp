// VerificationCode.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#include "VerificationCode.h"
#include "Segmentation.h"
#include "Recognition.h"


//#define DEBUG_LOG

CvANN_MLP mlp[2];
CvSVM svm;

Mat mlpResponse[2];
Mat Response;


//////////////////////////////////////////////////////////////////////////

int mlpRecognize(Mat& mat, char* code, float* conf);

int svmRecognize(Mat& mat, char* code);

float mlpGetSpecLabelConf(char label);

//////////////////////////////////////////////////////////////////////////


VERIFICATIONCODE_API int InitEngine()
{
	mlp[0].load("classifier\\modelm0.xml");
	mlp[1].load("classifier\\modelm1.xml");
	svm.load("classifier\\svm-0504-98.71-cv.xml");

	if (mlp[0].get_layer_count() <= 0 || mlp[1].get_layer_count() <= 0)
		return -1;

	if (svm.get_support_vector_count() <= 0)
		return -2;

	mlpResponse[0].create(1, GetClassCount(), CV_32FC1);
	mlpResponse[1].create(1, GetClassCount(), CV_32FC1);
	Response.create(1, GetClassCount(), CV_32FC1);

	return 1;
}


VERIFICATIONCODE_API int RecognizeCode(char* filePath, char* code, float* conf)
{
	memset(code, 0, CH_NUM * sizeof(char));
	memset(conf, 0, CH_NUM * sizeof(float));

	Mat img = imread(filePath, 4);

	if (img.data == NULL)
		return -1;

	Mat img_gray;
	cvtColor(img, img_gray, CV_BGR2GRAY);

	Mat img_bin;
	threshold(img_gray, img_bin, 128, 255, THRESH_BINARY);

	Mat img_mask = Mat::zeros(img.rows, img.cols, CV_8UC1);

	// remove image edge noise in img_mask, change 2 to 1.
	cv::Rect roi(2, 2, img.cols - 4, img.rows - 4);
	img_bin(roi).copyTo(img_mask(roi));

	// remove noise in img_mask by contour
	filter_bin_image(img_mask);

	// segment characters
	VCode vcode;

	if (segment_image(img, img_gray, img_mask, vcode) <= 0)
		return -1;

	Mat sq_img[CH_NUM];
	sq_img[0] = normalization(vcode._code[0]._img);
	sq_img[1] = normalization(vcode._code[1]._img);
	sq_img[2] = normalization(vcode._code[2]._img);
	sq_img[3] = normalization(vcode._code[3]._img);

#ifdef DEBUG_LOG
	imshow("img0", sq_img[0]);
	imshow("img1", sq_img[1]);
	imshow("img2", sq_img[2]);
	imshow("img3", sq_img[3]);
#endif

	const int MIN_LEN = 10;
	const int MAX_LEN = 40;

	for (int i = 0; i < CH_NUM; i++)
	{
		if (sq_img[i].rows < MIN_LEN || sq_img[i].rows > MAX_LEN)
			return -1;
	}

	const Size normSize(20, 20);
	char mlp_code[CH_NUM], svm_code[CH_NUM];

	int same_cnt = 0;
	for (int i = 0; i < CH_NUM; i++)
	{
		Mat imgNorm = NormalizeImage(sq_img[i], normSize);

		Mat mat = CalcSimpleFeature(imgNorm);
		Mat mat2 = CalcHogFeature(imgNorm);

		mlpRecognize(mat, &(mlp_code[i]), &(conf[i]));
		svmRecognize(mat2, &(svm_code[i]));

		if (mlp_code[i] == svm_code[i])
		{
			code[i] = mlp_code[i];
			same_cnt++;
		}
		else
		{
#ifdef DEBUG_LOG
			cout << i << " mlp: " << mlp_code[i] << "[" << conf[i] << "]" << " svm: " << svm_code[i] << endl;
#endif
			if (1/*conf[i] < 0.90f || conf[i] > 1.4f*/)
			{
				code[i] = svm_code[i];
				conf[i] = mlpGetSpecLabelConf(svm_code[i]);
			}
			else
			{
				code[i] = mlp_code[i];
			}
		}
	}

	return same_cnt;
}


VERIFICATIONCODE_API int ReleaseEngine()
{
	mlp[0].clear();
	mlp[1].clear();
	svm.clear();

	mlpResponse[0].release();
	mlpResponse[1].release();
	Response.release();

	return 1;
}


//////////////////////////////////////////////////////////////////////////


float mlpGetSpecLabelConf(char label)
{
	char buf[3] = {0};
	buf[0] = label;
	if (label >= 'a')
		buf[1] = label;

	int idx = MapLabel2Index(string(buf));

	float conf = 0.0f;
	if (idx > 0)
		conf = Response.at<float>(0, idx);

	return conf;
}


int mlpRecognize(Mat& mat, char* code, float* conf)
{
	mlp[0].predict(mat, mlpResponse[0]);
	mlp[1].predict(mat, mlpResponse[1]);
	Response = (mlpResponse[0] + mlpResponse[1]) / 2;

	Point maxLoc(0, 0);
	double maxVal = 0.0;
	minMaxLoc(Response, 0, &maxVal, 0, &maxLoc);

	string label = MapIndex2Label(maxLoc.x);

	*code = label[0];
	*conf = maxVal;

	return 1;
}


int svmRecognize(Mat& mat, char* code)
{
	int idx = (int)svm.predict(mat);

	string label = MapIndex2Label(idx);

	*code = label[0];

	return 1;
}

