#ifndef __RECOGNITION_H__
#define __RECOGNITION_H__

//////////////////////////////////////////////////////////////////////////

int GetClassCount();

string MapIndex2Label(int idx);

int MapLabel2Index(string label);

Mat NormalizeImage(Mat& imgSrc, Size normSize);

Mat CalcSimpleFeature(Mat& imgNorm);

Mat CalcHogFeature(Mat& imgNorm);

#endif