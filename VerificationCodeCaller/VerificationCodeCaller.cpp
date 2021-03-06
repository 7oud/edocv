// VerificationCodeCaller.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include "util.h"

#include "..\VerificationCode\VerificationCode.h"

#include <Windows.h>


//////////////////////////////////////////////////////////////////////////

void label_unclassified_images(string path);

//////////////////////////////////////////////////////////////////////////


int _tmain(int argc, _TCHAR* argv[])
{
	if (InitEngine() < 0)
	{
		cout << "init failed" << endl;
		return -1;
	}

	string image_path = "..\\image\\download_image\\test";

	vector<string> image_list;
	get_image_list(image_path, image_list);


	int lv_cnt[5] = {0}, fail_cnt = 0;
	int lv_correct_cnt[5] = {0};
	double totalTime = 0;

	for (vector<string>::iterator it = image_list.begin(); it != image_list.end(); ++it)
	{
		string img_file = image_path + "\\" + *it;

		Mat img = imread(img_file, 4);
		if (img.data == NULL)
		{
			DeleteFile(img_file.c_str());
			continue;
		}

		double t = (double)getTickCount();

		char code[4] = {0};
		float conf[4] = {0};
		int ret = RecognizeCode((char*)(img_file.c_str()), code, conf);

		t = (double)getTickCount() - t;
		totalTime += t;

		if (ret >= 0)
		{
			cout << "[" << ret << "]" << "\t" << *it << endl;
			cout << "\t" << code[0] << code[1] << code[2] << code[3];
			cout << "\t" << "[" << conf[0] << " " << conf[1] << " " << conf[2] << " " << conf[3] << "]" << endl;

			char str[5];
			str[0] = code[0];str[1] = code[1];str[2] = code[2];str[3] = code[3];str[4]='\0';

			if (it->substr(6, 4) == string(str))
				lv_correct_cnt[ret]++;
			else
			{
				//imshow("code", img);
				//waitKey(0);
				//destroyAllWindows();
			}
			lv_cnt[ret]++;
		}
		else
		{
			cout << *it << " ";
			cout << "[" << ret << "]" << "\t" << "pass." << endl;
			//MoveFile(img_file.c_str(), (image_path + "\\@\\" + *it).c_str());
			fail_cnt++;
		}

		imshow("code", img);
		waitKey(0);
		destroyAllWindows();
	}

	cout << "FAIL: " << fail_cnt << endl;

	int total_cnt = 0, correct_cnt = 0;
	for (int i = 0; i < 5; i++)
	{
		total_cnt += lv_cnt[i];
		correct_cnt += lv_correct_cnt[i];

		cout << "Lv[" << i << "]: " << lv_correct_cnt[i] << "/" << lv_cnt[i] << " = " << lv_correct_cnt[i]/(float)lv_cnt[i] << endl;
	}
	total_cnt += fail_cnt;

	cout << "TOTAL: " << correct_cnt << "/" << total_cnt << " = " << correct_cnt/(float)total_cnt << endl;
	cout << "Time : " << totalTime/(double)getTickFrequency()*1000. << "/" << total_cnt << " " << 
		totalTime/(double)getTickFrequency()*1000./total_cnt << endl;

	if (ReleaseEngine() < 0)
		cout << "release failed" << endl;

	return 0;
}


void label_unclassified_images(string image_path)
{
	vector<string> image_list;
	get_image_list(image_path, image_list);

	int count = 0;
	for (vector<string>::iterator it = image_list.begin(); it != image_list.end(); ++it)
	{
		Mat img = imread(image_path + "/" + *it);

		imshow("image", img);
		int key = waitKey(0);

		if (key == 27)
		{
			cout << "ERROR character, pass it" << endl;
			DeleteFile((image_path + "/" + *it).c_str());
			destroyAllWindows();
			continue;
		}

		char buf[8] = {0};
		buf[0] = (char)key;

		string label_folder = image_path + "/" + buf;
		mkdir(label_folder.c_str());

		if (MoveFile((image_path + "/" + *it).c_str(), (label_folder + "/" + *it).c_str()))
		{
			cout << "image " << count++ << ": ";
			cout << "move to folder [" << buf << "]" << endl;
			destroyAllWindows();
		}
	}
}

