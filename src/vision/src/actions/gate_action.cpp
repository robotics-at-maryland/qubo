#include "gate_action.h"

using namespace cv;
using namespace std;

GateAction::GateAction(){
	
	namedWindow("Canny output");
	moveWindow("Canny output", 20, 20);

	namedWindow("detected lines");
	moveWindow("detected lines", 20, 20);
	
	
};
GateAction::~GateAction(){};


void GateAction::updateAction(const Mat cframe){


	//sgillen@20171021-11:10 will need to move this to it's own function I think
	Mat dst, cdst;
	Canny(cframe, dst, 50, 200, 3);

	cvtColor(dst, cdst, CV_GRAY2BGR);
	vector<Vec4i> lines;
	HoughLinesP(dst, lines, 1, CV_PI/180, 50, 50, 10 );
	for( size_t i = 0; i < lines.size(); i++ )
		{
			Vec4i l = lines[i];
			line( cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);
		}

	
	//imshow("source", cframe);
	imshow("Canny output", dst);
	imshow("detected lines", cdst);
	

	waitKey(5);
	
}
