#include "stdafx.h"
#include "Segmentation.h"


double getThreshVal_Otsu_mask(const Mat& _src)
{
	Size size = _src.size();

	const int N = 256;
	int i, j, h[N] = {0}, count = 0;
	for( i = 0; i < size.height; i++ )
	{
		const uchar* src = _src.data + _src.step*i;
		for( j = 0; j < size.width; j++ )
		{
			if (src[j] != 0)
			{
				h[src[j]]++;
				count++;
			}
		}
	}

	double mu = 0, scale = 1./count;
	for( i = 0; i < N; i++ )
		mu += i*(double)h[i];

	mu *= scale;
	double mu1 = 0, q1 = 0;
	double max_sigma = 0, max_val = 0;

	for( i = 0; i < N; i++ )
	{
		double p_i, q2, mu2, sigma;

		p_i = h[i]*scale;
		mu1 *= q1;
		q1 += p_i;
		q2 = 1. - q1;

		if( std::min(q1,q2) < FLT_EPSILON || std::max(q1,q2) > 1. - FLT_EPSILON )
			continue;

		mu1 = (mu1 + i*p_i)/q1;
		mu2 = (mu - q1*mu1)/q2;
		sigma = q1*q2*(mu1 - mu2)*(mu1 - mu2);
		if( sigma > max_sigma )
		{
			max_sigma = sigma;
			max_val = i;
		}
	}

	return max_val;
}


int func_nc8(int *b)
{
	//端点的连通性检测
	int n_odd[4] = { 1, 3, 5, 7 };
	int i, j, sum, d[10];          

	for (i = 0; i <= 9; i++) 
	{
		j = i;
		if (i == 9) j = 1;
		if (abs(*(b + j)) == 1)
		{
			d[i] = 1;
		} 
		else 
		{
			d[i] = 0;
		}
	}
	sum = 0;
	for (i = 0; i < 4; i++)
	{
		j = n_odd[i];
		sum = sum + d[j] - d[j] * d[j + 1] * d[j + 2];
	}
	return (sum);
}

void cvHilditchThin(cv::Mat& src, cv::Mat& dst)
{
	if(src.type()!=CV_8UC1)
		return;

	if(dst.data!=src.data)
		src.copyTo(dst);

	//8邻域的偏移量
	int offset[9][2] = {{0,0},{1,0},{1,-1},{0,-1},{-1,-1},
	{-1,0},{-1,1},{0,1},{1,1} };
	//四邻域的偏移量
	int n_odd[4] = { 1, 3, 5, 7 };      
	int px, py;                        
	int b[9];                      //3*3格子的灰度信息
	int condition[6];              //1-6个条件是否满足
	int counter;                   //移去像素的数量
	int i, x, y, copy, sum;      

	uchar* img;
	int width, height;
	width = dst.cols;
	height = dst.rows;
	img = dst.data;
	int step = dst.step ;
	do
	{
		counter = 0;

		for (y = 0; y < height; y++)
		{
			for (x = 0; x < width; x++) 
			{
				for (i = 0; i < 9; i++) 
				{
					b[i] = 0;
					px = x + offset[i][0];
					py = y + offset[i][1];
					if (px >= 0 && px < width && py >= 0 && py <height) 
					{
						// printf("%d\n", img[py*step+px]);
						if (img[py*step+px] == 255)
						{
							b[i] = 1;
						} 
						else if (img[py*step+px]  == 128) 
						{
							b[i] = -1;
						}
					}
				}
				for (i = 0; i < 6; i++)
				{
					condition[i] = 0;
				}

				//条件1，是前景点
				if (b[0] == 1) condition[0] = 1;

				//条件2，是边界点
				sum = 0;
				for (i = 0; i < 4; i++) 
				{
					sum = sum + 1 - abs(b[n_odd[i]]);
				}
				if (sum >= 1) condition[1] = 1;

				//条件3， 端点不能删除
				sum = 0;
				for (i = 1; i <= 8; i++)
				{
					sum = sum + abs(b[i]);
				}
				if (sum >= 2) condition[2] = 1;

				//条件4， 孤立点不能删除
				sum = 0;
				for (i = 1; i <= 8; i++)
				{
					if (b[i] == 1) sum++;
				}
				if (sum >= 1) condition[3] = 1;

				//条件5， 连通性检测
				if (func_nc8(b) == 1) condition[4] = 1;

				//条件6，宽度为2的骨架只能删除1边
				sum = 0;
				for (i = 1; i <= 8; i++)
				{
					if (b[i] != -1) 
					{
						sum++;
					} else 
					{
						copy = b[i];
						b[i] = 0;
						if (func_nc8(b) == 1) sum++;
						b[i] = copy;
					}
				}
				if (sum == 8) condition[5] = 1;

				if (condition[0] && condition[1] && condition[2] &&condition[3] && condition[4] && condition[5])
				{
					img[y*step+x] = 128; //可以删除，置位128，128是删除标记，但该信息对后面像素的判断有用
					counter++;
					//printf("----------------------------------------------\n");
					//PrintMat(dst);
				}
			} 
		}

		if (counter != 0)
		{
			for (y = 0; y < height; y++)
			{
				for (x = 0; x < width; x++)
				{
					if (img[y*step+x] == 128)
						img[y*step+x] = 0;

				}
			}
		}

	}while (counter != 0);

}


void cvHilditchThin1(cv::Mat& src, cv::Mat& dst)
{
	if(src.type()!=CV_8UC1)
	{
		return;
	}

	if (dst.data!=src.data)
		src.copyTo(dst);

	int i, j;
	int width, height;
	//之所以减2，是方便处理8邻域，防止越界
	width = src.cols -2;
	height = src.rows -2;
	int step = src.step;
	int  p2,p3,p4,p5,p6,p7,p8,p9;
	uchar* img;
	bool ifEnd;
	int A1;
	cv::Mat tmpimg;
	while(1)
	{
		dst.copyTo(tmpimg);
		ifEnd = false;
		img = tmpimg.data+step;
		for(i = 2; i < height; i++)
		{
			img += step;
			for(j =2; j<width; j++)
			{
				uchar* p = img + j;
				A1 = 0;
				if( p[0] > 0)
				{
					//p2,p3 01模式
					if(p[-step]==0&&p[-step+1]>0)	A1++;
					//p3,p4 01模式
					if(p[-step+1]==0&&p[1]>0)		A1++;
					//p4,p5 01模式
					if(p[1]==0&&p[step+1]>0)		A1++;
					//p5,p6 01模式
					if(p[step+1]==0&&p[step]>0)		A1++;
					//p6,p7 01模式
					if(p[step]==0&&p[step-1]>0)		A1++;
					//p7,p8 01模式
					if(p[step-1]==0&&p[-1]>0)		A1++;
					//p8,p9 01模式
					if(p[-1]==0&&p[-step-1]>0)		A1++;
					//p9,p2 01模式
					if(p[-step-1]==0&&p[-step]>0)	A1++;

					p2 = p[-step]>0?1:0;
					p3 = p[-step+1]>0?1:0;
					p4 = p[1]>0?1:0;
					p5 = p[step+1]>0?1:0;
					p6 = p[step]>0?1:0;
					p7 = p[step-1]>0?1:0;
					p8 = p[-1]>0?1:0;
					p9 = p[-step-1]>0?1:0;
					//计算AP2,AP4
					int A2 = 0, A4 = 0;
					//if(p[-step]>0)
					{
						if(p[-2*step]==0&&p[-2*step+1]>0)	A2++;
						if(p[-2*step+1]==0&&p[-step+1]>0)	A2++;
						if(p[-step+1]==0&&p[1]>0)			A2++;
						if(p[1]==0&&p[0]>0)					A2++;
						if(p[0]==0&&p[-1]>0)					A2++;
						if(p[-1]==0&&p[-step-1]>0)			A2++;
						if(p[-step-1]==0&&p[-2*step-1]>0)	A2++;
						if(p[-2*step-1]==0&&p[-2*step]>0)	A2++;
					}

					//if(p[1]>0)
					{
						if(p[-step+1]==0&&p[-step+2]>0)		A4++;
						if(p[-step+2]==0&&p[2]>0)			A4++;
						if(p[2]==0&&p[step+2]>0)			A4++;
						if(p[step+2]==0&&p[step+1]>0)		A4++;
						if(p[step+1]==0&&p[step]>0)			A4++;
						if(p[step]==0&&p[0]>0)				A4++;
						if(p[0]==0&&p[-step]>0)				A4++;
						if(p[-step]==0&&p[-step+1]>0)		A4++;
					}

					if((p2+p3+p4+p5+p6+p7+p8+p9)>1 && (p2+p3+p4+p5+p6+p7+p8+p9)<7 && A1==1)
					{
						if(((p2==0||p4==0||p8==0)||A2!=1)&&((p2==0||p4==0||p6==0)||A4!=1)) 
						{
							dst.at<uchar>(i,j) = 0; //满足删除条件，设置当前像素为0
							ifEnd = true;
						}
					}
				}
			}
		}
		//如果两个子迭代已经没有可以细化的像素了，则退出迭代
		if(!ifEnd) 
			break;
	}
}


void extend_border(Mat& img, int pix)
{
	Mat ext = Mat::zeros(img.rows+2*pix, img.cols+2*pix, CV_8UC1);
	Mat roi = ext(Rect(pix, pix, img.cols, img.rows));
	img.copyTo(roi);
	img = ext;

	return ;
}


void shrink_border(Mat& img, int pix)
{
	int _rows = img.rows-2*pix;
	int _cols = img.cols-2*pix;
	if (_rows <= 0 || _cols <= 0)
		return ;

	Mat shr(_rows, _cols, CV_8UC1);
	
	Rect rc(pix, pix, _cols, _rows);
	img(rc).copyTo(shr);

	img = shr;
}


int filter_bin_image(Mat& bin_img, Rect& loc, int w, int h, double a)
{
	vector<Vec4i> hierarchy;
	vector<vector<Point> > contours;

	Mat img_contour;
	bin_img.copyTo(img_contour);

	findContours(img_contour, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

	int count = 0;
	for(int idx = 0; idx >= 0 && contours.size() > 0; idx = hierarchy[idx][0])
	{
		const vector<Point>& c = contours[idx];
		Mat contour = Mat(c);

		cv::Rect rc = boundingRect(contour);
		double area = fabs(contourArea(contour));

		rc.x = MAX(0, rc.x - 1);
		rc.y = MAX(0, rc.y - 1);

		// filter small noise
		if (rc.width <= w || rc.height <= h || area < a)
		{
			for (int i = rc.y; i <= rc.y + rc.height; i++)
			{
				for (int j = rc.x; j <= rc.x + rc.width; j++)
				{
					uchar* ptr = &((uchar*)(bin_img.data + bin_img.step * i))[j];
					*ptr = 0;
				}
			}
		}
		else
		{
			loc = rc;
			count++;
		}
	}

	return count;
}


Mat mask_image(Mat& img, Mat& img_mask)
{
	Mat img_show;
	img.copyTo(img_show);

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			uchar* ptr_mask = &((uchar*)(img_mask.data + img_mask.step * i))[j];
			uchar* ptr = &((uchar*)(img_show.data + img_show.step * i))[j*3];

			if (*ptr_mask == 0)
			{
				ptr[0] = ptr[1] = ptr[2] = 0;
			}
		}
	}

	return img_show;
}



void show_hist(Mat src, Mat gray, Mat hsv, Mat mask)
{
	Mat dst;

	/// Separate the image in 3 places ( B, G and R )
	vector<Mat> bgr_planes, hsv_planes;
	split( src, bgr_planes );
	split( hsv, hsv_planes );

	/// Establish the number of bins
	int histSize = 256;

	/// Set the ranges ( for B,G,R) )
	float range[] = { 0, 256 } ;
	const float* histRange = { range };

	bool uniform = true; bool accumulate = false;

	Mat hsv_hist, gray_hist, b_hist, g_hist, r_hist;

	/// Compute the histograms:
	calcHist( &gray, 1, 0, Mat(), gray_hist, 1, &histSize, &histRange, uniform, accumulate );
	calcHist( &hsv_planes[0], 1, 0, Mat(), hsv_hist, 1, &histSize, &histRange, uniform, accumulate );

	calcHist( &bgr_planes[0], 1, 0, mask, b_hist, 1, &histSize, &histRange, uniform, accumulate );
	calcHist( &bgr_planes[1], 1, 0, mask, g_hist, 1, &histSize, &histRange, uniform, accumulate );
	calcHist( &bgr_planes[2], 1, 0, mask, r_hist, 1, &histSize, &histRange, uniform, accumulate );

	// Draw the histograms for B, G and R
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound( (double) hist_w/histSize );

	Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );
	Mat gray_histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );
	Mat hsv_histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

	/// Normalize the result to [ 0, histImage.rows ]
	normalize(hsv_hist, hsv_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
	normalize(gray_hist, gray_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

	/// Draw for each channel
	for( int i = 1; i < histSize; i++ )
	{
		line( histImage, Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) ,
			Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
			Scalar( 255, 0, 0), 2, 8, 0  );
		line( histImage, Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ) ,
			Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
			Scalar( 0, 255, 0), 2, 8, 0  );
		line( histImage, Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ) ,
			Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
			Scalar( 0, 0, 255), 2, 8, 0  );

		line( gray_histImage, Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) ,
			Point( bin_w*(i), hist_h - cvRound(gray_hist.at<float>(i)) ),
			Scalar( 128, 128, 128), 2, 8, 0  );

		line( hsv_histImage, Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) ,
			Point( bin_w*(i), hist_h - cvRound(hsv_hist.at<float>(i)) ),
			Scalar( 128, 128, 128), 2, 8, 0  );
	}

	namedWindow("calcHist Demo", WINDOW_AUTOSIZE );
	imshow("calcHist Demo", histImage );
	imshow("gray_histImage", gray_histImage);
	imshow("hsv_histImage", hsv_histImage);

}


Mat show_segment_image(const Mat& img, const Mat& img_mask)
{
	vector<Vec4i> hierarchy;
	vector<vector<Point> > contours;

	Mat img_contour;
	img_mask.copyTo(img_contour);

	findContours(img_contour, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);	//CV_RETR_EXTERNAL

	Mat sum_img = Mat::zeros(img.size(), CV_8UC3);
	for(int k = 0, idx = 0; idx >= 0; idx = hierarchy[idx][0], k++)
	{
		const vector<Point>& c = contours[idx];
		Mat contour = Mat(c);

		cv::Rect rc = boundingRect(contour);

		Mat fill = Mat::zeros(img.rows, img.cols, CV_8UC3);

		Scalar color = CV_RGB(0,0,0);
		switch (k)
		{
		case 0:	color = CV_RGB(255,0,0);	break;
		case 1:	color = CV_RGB(0,255,0);	break;
		case 2:	color = CV_RGB(0,0,255);	break;
		case 3:	color = CV_RGB(255,255,255);	break;
		}
		drawContours( fill, contours, idx, color, CV_FILLED, 8, hierarchy );

		sum_img += fill;
	}

	return sum_img;
}

Rect boundingRectOfMaskImage(const Mat& image)
{
	int min_x = image.cols;
	int min_y = image.rows;
	int max_x = 0;
	int max_y = 0;
	for (int j = 0; j < image.rows; j++) 
	{
		for (int i = 0; i < image.cols; i++) 
		{
			if (image.at<uchar>(j, i) != 0) 
			{
				min_x = std::min(min_x, i);
				min_y = std::min(min_y, j);
				max_x = std::max(max_x, i);
				max_y = std::max(max_y, j);
			}
		}
	}

	return Rect(min_x, min_y, std::max(max_x - min_x + 1, 0), std::max(max_y - min_y + 1, 0));
}


void get_2_masks_by_h_channel(Mat& img_rgb, Mat& img_h_bin1, Mat& img_h_bin2)
{
	Mat img_hsv, img_h;
	cvtColor(img_rgb, img_hsv, CV_BGR2HSV);

	vector<Mat> hsv_planes;
	split(img_hsv, hsv_planes);

	img_h = hsv_planes[0];

	int thre = getThreshVal_Otsu_mask(img_h);
	threshold(img_h, img_h_bin1, thre, 255, THRESH_BINARY);

	img_h_bin2 = Mat::zeros(img_h.rows, img_h.cols, CV_8UC1);
	for (int i = 0; i < img_h.rows; i++)
	{
		for (int j = 0; j < img_h.cols; j++)
		{
			uchar* ptr_h = &((uchar*)(img_h.data + img_h.step * i))[j];
			uchar* ptr_hb = &((uchar*)(img_h_bin2.data + img_h_bin2.step * i))[j];

			if (*ptr_h > 0 && *ptr_h < thre)
				*ptr_hb = 255;
		}
	}
}


int check_code_valid(const VCode& vcode, int thre1)
{
	for (int i = 0; i < CH_NUM; i++)
	{
		if (vcode._code[i]._img.rows < thre1 && vcode._code[i]._img.cols < thre1)
			return -1;
	}

	return 1;
}


int sort_location(Code arr[], int size, VCode& vcode)
{
	vector<pair<int, int> > v;
	for (int i = 0; i < size; i++)
		v.push_back(make_pair(arr[i]._rc.x, i));

	sort(v.begin(), v.end());

	for (int i = 0; i < v.size(); i++)
		vcode._code[i] = arr[v[i].second];

	return 0;
}


int process_2ch_adhere(const Mat& img, Mat& img_mask, int count, VCode& vcode)
{
	// find adhered character
	int max_w = 0, max_w_idx = -1;
	for (int i = 0; i < count; i++)
	{
		if (vcode._code[i]._rc.width > max_w)
		{
			max_w = vcode._code[i]._rc.width;
			max_w_idx = i;
		}
	}

	if (vcode._code[max_w_idx]._rc.width < 1.2 * vcode._code[max_w_idx]._rc.height)
		return -1;

	Rect r = vcode._code[max_w_idx]._rc;
	Mat rgb = vcode._code[max_w_idx]._img;

	Mat h_bin1, h_bin2;
	get_2_masks_by_h_channel(rgb, h_bin1, h_bin2);

	extend_border(h_bin1);
	extend_border(h_bin2);

	Rect rect1, rect2;
 	int num1 = filter_bin_image(h_bin1, rect1, 4, 4, 3);
 	int num2 = filter_bin_image(h_bin2, rect2, 4, 4, 3);

	shrink_border(h_bin1);
	shrink_border(h_bin2);

	if (num1 != 1 || num2 != 1)
		return -1;

	Rect left_rc, right_rc;
	Mat left_half, right_half;

	if (rect1.x < rect2.x)
	{
		left_rc = Rect(0,0,rect1.x+rect1.width,h_bin1.rows);
		right_rc = Rect(rect2.x,0,h_bin2.cols-rect2.x,h_bin2.rows);
		left_half = h_bin1(left_rc);
		right_half = h_bin2(right_rc);
	}
	else
	{
		left_rc = Rect(0,0,rect2.x+rect2.width,h_bin2.rows);
		right_rc = Rect(rect1.x,0,h_bin1.cols-rect1.x,h_bin1.rows);
		left_half = h_bin2(left_rc);
		right_half = h_bin1(right_rc);
	}

	Mat _img1 = mask_image(rgb(left_rc), left_half);
	Mat _img2 = mask_image(rgb(right_rc), right_half);

	Rect rr1 = boundingRectOfMaskImage(left_half);
	Rect rr2 = boundingRectOfMaskImage(right_half);

	Code code[CH_NUM];
	code[0]._rc = Rect(r.x + left_rc.x + rr1.x, r.y + left_rc.y + rr1.y, rr1.width, rr1.height);
	code[0]._img = _img1(rr1);
	code[1]._rc = Rect(r.x + right_rc.x + rr2.x, r.y + right_rc.y + rr2.y, rr2.width, rr2.height);
	code[1]._img = _img2(rr2);

	int start_idx = 2;
	for (int i = 0; i < count; i++)
	{
		if (i != max_w_idx)
		{
			code[start_idx] = vcode._code[i];
			start_idx++;
		}
	}

	VCode sort_code;
	sort_location(code, CH_NUM, sort_code);

	if (check_code_valid(sort_code, 9) < 0)
		return -1;

	vcode = sort_code;

	return 2;
}

int process_0ch_adhere(const Mat& img, Mat& img_mask, int count, VCode& vcode)
{
	VCode sort_code;
	sort_location(vcode._code, CH_NUM, sort_code);

	if (check_code_valid(sort_code, 6) < 0)
		return -1;

	vcode = sort_code;

	return 1;
}

int segment_image(const Mat& img, Mat& img_gray, Mat& img_mask, VCode& vcode)
{
	vector<Vec4i> hierarchy;
	vector<vector<Point> > contours;

	Mat img_contour;
	img_mask.copyTo(img_contour);

	findContours(img_contour, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);	//CV_RETR_EXTERNAL

	int num = 0;
	for(int idx = 0; idx >= 0; idx = hierarchy[idx][0])
	{
		const vector<Point>& c = contours[idx];
		Mat contour = Mat(c);

		cv::Rect rc = boundingRect(contour);

		Mat fill = Mat::zeros(img.rows, img.cols, CV_8UC3);
		drawContours( fill, contours, idx, CV_RGB(0,0,255), CV_FILLED, 8, hierarchy );

		if (num < CH_NUM)
		{
			img(rc).copyTo(vcode._code[num]._img);
			Mat rc_fill = fill(rc);

			vcode._code[num]._rc = rc;
			for (int i = 0; i < rc.height; i++)
			{
				for (int j = 0; j < rc.width; j++)
				{
					uchar* ptr_fill = &((uchar*)(rc_fill.data + rc_fill.step * i))[j*3];
					uchar* ptr_img = &((uchar*)(vcode._code[num]._img.data + vcode._code[num]._img.step * i))[j*3];
					if (ptr_fill[0] == 0)
						ptr_img[0] = ptr_img[1] = ptr_img[2] = 0;
				}
			}
			num++;
		}
	}

	int ret = 0;
	if (num == CH_NUM)
	{
		ret = process_0ch_adhere(img, img_mask, num, vcode);
	}
	else if (num == CH_NUM - 1)
	{
		ret = process_2ch_adhere(img, img_mask, num, vcode);
	}
	else
	{
		ret = -2;
	}

	return ret;
}


Mat normalization(Mat& img)
{
	int offset_x = 0.2f * img.cols;
	int offset_y = 0.2f * img.rows;

	Size normed_sz(img.cols + 2 * offset_x, img.rows + 2 * offset_y);

	Mat normed_img = Mat::zeros(normed_sz, CV_8UC3);
	Mat roi_img = normed_img(Rect(offset_x, offset_y, img.cols, img.rows));
	img.copyTo(roi_img);

	Mat square_img;
	if (normed_img.cols > normed_img.rows)
	{
		int off = normed_img.cols - normed_img.rows;
		square_img = Mat::zeros(normed_img.cols, normed_img.cols, CV_8UC3);
		roi_img = square_img(Rect(0, off/2, normed_img.cols, normed_img.rows));
		normed_img.copyTo(roi_img);

	}
	else
	{
		int off = normed_img.rows - normed_img.cols;
		square_img = Mat::zeros(normed_img.rows, normed_img.rows, CV_8UC3);
		roi_img = square_img(Rect(off/2, 0, normed_img.cols, normed_img.rows));
		normed_img.copyTo(roi_img);
	}

	return square_img;
}
