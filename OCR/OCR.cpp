// OCR.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "Util.h"



//////////////////////////////////////////////////////////////////////////

int TrainANN(string classifier_file);
int TestANN(string classifier_file);
int TrainSVM(string classifier_file);
int TestSVM(string classifier_file);

int AddSample();

int FusedMlpPredict();

//////////////////////////////////////////////////////////////////////////

const string trainImgPath = "D:\\WorkSpace\\Work4\\VerificationCode\\bin\\image\\样本\\";
const string testImgPath = "D:\\WorkSpace\\Work4\\VerificationCode\\bin\\image\\segment_image\\";

const Size normSize(20, 20);

#define EVAL_TRAINING_SAMPLE

/* ANN下置信度为1.403105时，response矩阵的每个值都是1.403105，需要分析原因 */


int _tmain(int argc, _TCHAR* argv[])
{
	//AddSample();

	//TrainANN("ann.xml");
	//TestANN("ann-95.26.xml");

	TrainSVM("svm.xml");
	//TestSVM("svm.xml");

	//FusedMlpPredict();

	return 0;
}


int TrainANN(string classifier_file)
{
	cout << "Init file list..." << endl;

	vector<LabeledSampleList> trainSamplesList;
	GetFilesList(trainImgPath, "BMP", trainSamplesList);

	int clsCnt = trainSamplesList.size();

#ifndef EVAL_TRAINING_SAMPLE 

	Mat matTrainData, matTrainResponse;
	makeTrainMatrix(trainImgPath, trainSamplesList, normSize, matTrainData, matTrainResponse);

	const int midLayerNeuronCnt = 60;
	int layerSz[] = {matTrainData.cols, midLayerNeuronCnt, clsCnt};
	Mat matLayers(1, (int)(sizeof(layerSz)/sizeof(layerSz[0])), CV_32S, layerSz);

	cout << endl << "Start Training, Wait!" << endl;

	CvANN_MLP_TrainParams params;  
	params.train_method = CvANN_MLP_TrainParams::RPROP;  
	params.bp_dw_scale = 0.1;  
	params.bp_moment_scale = 0.1;  
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 20000, 0.0000001);

	CvANN_MLP mlp;
	mlp.create(matLayers);
	int iterCnt = mlp.train(matTrainData, matTrainResponse, Mat(), Mat(), params);

	cout << iterCnt << ": Training finished!" << endl;

	mlp.save(classifier_file.c_str());

#else

	CvANN_MLP mlp;
	mlp.load(classifier_file.c_str());

#endif

	cout << "Evaluate training samples..." << endl;

	int correct = 0, total = 0;
	double totalTime = 0;
	Mat matRecogResponse(1, clsCnt, CV_32FC1);

	for (size_t i = 0; i < trainSamplesList.size(); i++)
	{
		for (size_t j = 0; j < trainSamplesList[i].fileList.size(); j++)
		{
			string sampleName = trainImgPath + "\\" + trainSamplesList[i].label + "\\" + trainSamplesList[i].fileList[j];

			Mat img = imread(sampleName, 4);
			if (img.data == NULL)
				continue;

			double t = (double)getTickCount();

			Mat mat = CalcImageFeature(img, normSize);

			mlp.predict(mat, matRecogResponse);

			Point maxLoc(0, 0);
			double maxVal = 0.0;
			minMaxLoc(matRecogResponse, 0, &maxVal, 0, &maxLoc);
			string label = MapIndex2Label(maxLoc.x);

			t = (double)getTickCount() - t;
			totalTime += t;

			//cout << label << " " << trainSamplesList[i].label << endl;
			//imshow("img", img);
			//waitKey(0);

			if (label == trainSamplesList[i].label)
			{
				correct++;
			}
			else
			{
				if (0)
				{
					string folder = "train-error-ann\\";
					int ret = mkdir(folder.c_str());
					folder = folder + "\\" + label;

					mkdir(folder.c_str());

					char name[256];
					//sprintf(name, "%s\\%s_%f-%s", folder.c_str(), label.c_str(), maxVal, trainSamplesList[i].fileList[j].c_str());
					sprintf(name, "%s\\%s__%s__%f.bmp", folder.c_str(), trainSamplesList[i].fileList[j].c_str(), trainSamplesList[i].label.c_str(), maxVal);

					imwrite(name, img);
				}
			}

			total++;
		}
	}

	cout << "CORRECT: " << correct << "/" << total << " = " << correct/(float)total << endl;
	cout << "Time : " << totalTime/(double)getTickFrequency()*1000. << "/" << total << " " << 
		totalTime/(double)getTickFrequency()*1000./total << endl;

	return 0;
}


int TestANN(string classifier_file)
{
	CvANN_MLP mlp;
	mlp.load(classifier_file.c_str());

	vector<string> testSamplesList;
	GetFilesList(testImgPath, "BMP", testSamplesList);

	Mat matRecogResponse(1, GetClassCount(), CV_32FC1);
	for (size_t i = 0; i < testSamplesList.size(); i++)
	{
		string sampleName = testImgPath + testSamplesList[i];

		Mat img = imread(sampleName, 4);
		if (img.data == NULL)
			continue;

		Mat mat = CalcImageFeature(img, normSize);

		mlp.predict(mat, matRecogResponse);

		Point maxLoc(0, 0);
		double maxVal = 0.0;
		minMaxLoc(matRecogResponse, 0, &maxVal, 0, &maxLoc);

		string label = MapIndex2Label(maxLoc.x);

		//cout << label << ": " << maxVal << endl;
		//imshow("img", img);
		//waitKey(0);
		//destroyAllWindows();

		string folder = testImgPath + "\\" + label;

		mkdir(folder.c_str());

		char name[256];
		sprintf(name, "%s\\%d_%f.bmp", folder.c_str(), i, maxVal);
		imwrite(name, img);
		DeleteFile(sampleName.c_str());
	}

	return 0;
}


int TrainSVM(string classifier_file)
{
	cout << "Init file list..." << endl;

	vector<LabeledSampleList> trainSamplesList;
	GetFilesList(trainImgPath, "BMP", trainSamplesList);

	int clsCnt = trainSamplesList.size();

#ifndef EVAL_TRAINING_SAMPLE 

	// shuffle ?
	// scale ?
	// cross validation

	Mat matTrainData, matTrainResponse;
	makeSVMTrainMatrix(trainImgPath, trainSamplesList, normSize, matTrainData, matTrainResponse);

	cout << endl << "Start Training, Wait!" << endl;

	CvSVM svm;
	CvSVMParams param;   
	CvTermCriteria criteria;  

	//criteria= cvTermCriteria(CV_TERMCRIT_EPS, 1000, FLT_EPSILON);   
	criteria= cvTermCriteria(CV_TERMCRIT_EPS, 1000, 0.01);

	param = CvSVMParams(CvSVM::C_SVC, CvSVM::LINEAR, 10.0, 8.0, 1.0, 10.0, 0.5, 0.1, NULL, criteria);   //CvSVM::RBF

	//svm.train(matTrainData, matTrainResponse, Mat(), Mat(), param);
	svm.train_auto(matTrainData, matTrainResponse, Mat(), Mat(), param, 10);

	cout << "Training finished!" << endl;

	svm.save(classifier_file.c_str());
#else
	CvSVM svm;
	svm.load(classifier_file.c_str());
#endif

	cout << "Evaluate training samples..." << endl;

	int correct = 0, total = 0;
	double totalTime = 0;
	for (size_t i = 0; i < trainSamplesList.size(); i++)
	{
		for (size_t j = 0; j < trainSamplesList[i].fileList.size(); j++)
		{
			string sampleName = trainImgPath + "\\" + trainSamplesList[i].label + "\\" + trainSamplesList[i].fileList[j];

			Mat img = imread(sampleName, 4);
			if (img.data == NULL)
				continue;

			double t = (double)getTickCount();

			Mat mat = CalcImageFeature(img, normSize);

			int idx = (int)svm.predict(mat);
			string label = MapIndex2Label(idx);

			t = (double)getTickCount() - t;
			totalTime += t;

			//cout << label << " " << trainSamplesList[i].label << endl;
			//imshow("img", img);
			//waitKey(0);

			if (label == trainSamplesList[i].label)
				correct++;
			else
			{
				if (0)
				{
					string folder = "train-error-svm\\";
					int ret = mkdir(folder.c_str());
					folder = folder + "\\" + label;

					mkdir(folder.c_str());

					char name[256];
					sprintf(name, "%s\\%s__%s.bmp", folder.c_str(), trainSamplesList[i].fileList[j].c_str(), trainSamplesList[i].label.c_str());
					imwrite(name, img);
				}
			}

			total++;
		}
	}

	cout << "CORRECT: " << correct << "/" << total << " = " << correct/(float)total << endl;
	cout << "Time : " << totalTime/(double)getTickFrequency()*1000. << "/" << total << " " << 
		totalTime/(double)getTickFrequency()*1000./total << endl;

	return 0;
}


int TestSVM(string classifier_file)
{
	CvSVM svm;
	svm.load(classifier_file.c_str());

	vector<string> testSamplesList;
	GetFilesList(testImgPath, "BMP", testSamplesList);

	Mat matRecogResponse(1, GetClassCount(), CV_32FC1);
	for (size_t i = 0; i < testSamplesList.size(); i++)
	{
		string sampleName = testImgPath + testSamplesList[i];

		Mat img = imread(sampleName, 4);
		if (img.data == NULL)
			continue;

		Mat mat = CalcImageFeature(img, normSize);

		int idx = (int)svm.predict(mat);

		string label = MapIndex2Label(idx);

		//cout << label << endl;
		//imshow("img", img);
		//waitKey(0);
		//destroyAllWindows();

		string folder;
		if (label[0] > 'Z')
			folder = testImgPath + "\\" + label + label;
		else
			folder = testImgPath + "\\" + label;

		mkdir(folder.c_str());

		char name[256];
		sprintf(name, "%s\\%d.bmp", folder.c_str(), i);
		imwrite(name, img);
		DeleteFile(sampleName.c_str());
	}

	return 0;
}


int AddSample()
{
#if 1
	string path = "D:\\WorkSpace\\Work4\\VerifyCode\\bin\\样本\\";
	for (int i = 0; i < GetClassCount(); i++)
	{
		string label = MapIndex2Label(i);
		string path1 = path + "_" + label + "\\";
		string path2 = path + label + "\\";

		bool ret = FindIdenticalImage(path1, path2);
	}
#else
	string path = "D:\\WorkSpace\\Work4\\VerifyCode\\bin\\image\\segment_image\\";
	for (int i = 0; i < GetClassCount(); i++)
	{
		string label = MapIndex2Label(i);
		cout << label << endl << endl;

		string check_path = path + label + "\\";

		CheckIdenticalImage(check_path);
	}
#endif

	return 0;
}



int FusedMlpPredict()
{
	cout << "Init file list..." << endl;

	vector<LabeledSampleList> trainSamplesList;
	GetFilesList(trainImgPath, "BMP", trainSamplesList);

	int clsCnt = trainSamplesList.size();

	CvANN_MLP mlp[2];
	mlp[0].load("ann-0428-96.33.xml");
	mlp[1].load("ann-0429-96.69.xml");
	
	cout << "Evaluate training samples..." << endl;

	int correct = 0, total = 0;
	double totalTime = 0;
	Mat matResponse[2];
	Mat matRecogResponse;
	matResponse[0].create(1, clsCnt, CV_32FC1);
	matResponse[1].create(1, clsCnt, CV_32FC1);

	for (size_t i = 0; i < trainSamplesList.size(); i++)
	{
		for (size_t j = 0; j < trainSamplesList[i].fileList.size(); j++)
		{
			string sampleName = trainImgPath + "\\" + trainSamplesList[i].label + "\\" + trainSamplesList[i].fileList[j];

			Mat img = imread(sampleName, 4);
			if (img.data == NULL)
				continue;

			double t = (double)getTickCount();

			Mat mat = CalcImageFeature(img, normSize);

			mlp[0].predict(mat, matResponse[0]);
			mlp[1].predict(mat, matResponse[1]);

			matRecogResponse = (matResponse[0] + matResponse[1])/2;

			Point maxLoc(0, 0);
			double maxVal = 0.0;
			minMaxLoc(matRecogResponse, 0, &maxVal, 0, &maxLoc);
			string label = MapIndex2Label(maxLoc.x);

			t = (double)getTickCount() - t;
			totalTime += t;

			if (label == trainSamplesList[i].label)
			{
				correct++;
			}
			else
			{
				if (0)
				{
					string folder = "train-error-ann\\";
					int ret = mkdir(folder.c_str());
					folder = folder + "\\" + label;

					mkdir(folder.c_str());

					char name[256];
					//sprintf(name, "%s\\%s_%f-%s", folder.c_str(), label.c_str(), maxVal, trainSamplesList[i].fileList[j].c_str());
					sprintf(name, "%s\\%s__%s__%f.bmp", folder.c_str(), trainSamplesList[i].fileList[j].c_str(), trainSamplesList[i].label.c_str(), maxVal);

					imwrite(name, img);
				}
			}

			total++;
		}
	}

	cout << "CORRECT: " << correct << "/" << total << " = " << correct/(float)total << endl;
	cout << "Time : " << totalTime/(double)getTickFrequency()*1000. << "/" << total << " " << 
		totalTime/(double)getTickFrequency()*1000./total << endl;

	return 0;
}

