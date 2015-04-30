#ifndef __SEGMENTATION_H__
#define __SEGMENTATION_H__

#define CH_NUM 4

typedef struct
{
	Rect	_rc;
	Mat		_img;
}Code;

class VCode
{
public:
	VCode() 
	{
		for (int i = 0; i < CH_NUM; i++)
		{
			_code[i]._rc = Rect(0, 0, 0, 0);
			_code[i]._img = Mat();
		}
	}
	~VCode() {}
public:
	Code	_code[CH_NUM];
};


double getThreshVal_Otsu_mask(const Mat& _src);

void cvHilditchThin(cv::Mat& src, cv::Mat& dst);
void cvHilditchThin1(cv::Mat& src, cv::Mat& dst);

int segment_image(const Mat& img, Mat& img_gray, Mat& img_mask, VCode& vcode);

int filter_bin_image(Mat& bin_img, Rect& rc = Rect(), int w = 2, int h = 2, double a = 2.0);

Mat mask_image(Mat& img, Mat& img_mask);

void show_hist(Mat src, Mat gray, Mat hsv, Mat mask);

Mat show_segment_image(const Mat& img, const Mat& img_mask);

void extend_border(Mat& img, int pix = 1);
void shrink_border(Mat& img, int pix = 1);

Mat normalization(Mat& img);


#endif