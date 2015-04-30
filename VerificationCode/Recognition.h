#ifndef __RECOGNITION_H__
#define __RECOGNITION_H__

//////////////////////////////////////////////////////////////////////////

int GetClassCount();

string MapIndex2Label(int idx);

int MapLabel2Index(string label);

Mat CalcImageFeature(Mat& imgSrc, Size normSize);


#endif