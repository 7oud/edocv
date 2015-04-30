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

	//string image_path = "..\\image";
	string image_path = "..\\bin\\image\\download_image\\same";

	vector<string> image_list;
	get_image_list(image_path, image_list);

	int total_cnt = 0, correct_cnt = 0, idx = 0;
	for (vector<string>::iterator it = image_list.begin(); it != image_list.end(); ++it)
	{
		string img_file = image_path + "\\" + *it;

		Mat img = imread(img_file, 4);

		if (img.data == NULL)
		{
			DeleteFile(img_file.c_str());
			continue;
		}

		char code[4];
		float conf[4];
		int ret = RecognizeCode((char*)(img_file.c_str()), code, conf);

		if (ret < 0)
		{
			cout << "pass." << endl;
			//MoveFile(img_file.c_str(), (image_path + "\\@\\" + *it).c_str());
		}
		else
		{
			cout << *it << endl;
			cout << "\t" << code[0] << code[1] << code[2] << code[3] << endl;
			cout << "\t" << conf[0] << " " << conf[1] << " " << conf[2] << " " << conf[3] << endl;

			if (ret == 4)
			{
				cout << "\t" << " same" << endl;
			}
			else
			{
				cout << "\t" << " diff" << endl;
			}
		}

		//imshow("code", img);
		//waitKey(0);
		//destroyAllWindows();

		char str[5];
		str[0] = code[0];str[1] = code[1];str[2] = code[2];str[3] = code[3];str[4]='\0';
		if (it->substr(6, 4) == string(str))
		{
			correct_cnt++;
		}
		else
		{
			//imshow("code", img);
			//waitKey(0);
			//destroyAllWindows();
		}
				
		total_cnt++;
		idx++;
	}

	cout << correct_cnt << "/" << total_cnt << " = " << correct_cnt/(float)total_cnt << endl;

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
