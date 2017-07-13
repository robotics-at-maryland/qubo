#include "gate_action.h"

using namespace cv;
using namespace std;

GateAction::GateAction(){

	m_kernel_size = 5; //must be odd and positive
	m_canny_thresh = 10; //we may have to make this adaptive
	m_hough_thresh = 20;
};
GateAction::~GateAction(){};


void GateAction::updateAction(const Mat cframe){


	//sgillen@20171021-11:10 will need to move this to it's own function I think
	Mat dst, cdst;
	
	GaussianBlur(cframe, dst, Size( m_kernel_size, m_kernel_size ), 0, 0 );
		
	Canny(dst, dst, m_canny_thresh, m_canny_thresh*3, 3);

	cvtColor(dst, cdst, CV_GRAY2BGR);
	vector<Vec4i> lines;

	HoughLines(dst, lines, 1, CV_PI/180, m_hough_thresh, 50, 10 );
	
	for( size_t i = 0; i < lines.size(); i++ )
		{
			float rho = lines[i][0], theta = lines[i][1];
			//printf("line[%lu] = %f, %f \n", i, lines[i][0], lines[i][1]);
			Point pt1, pt2;
			double a = cos(theta), b = sin(theta);
			double x0 = a*rho, y0 = b*rho;
			pt1.x = cvRound(x0 + 1000*(-b));
			pt1.y = cvRound(y0 + 1000*(a));
			pt2.x = cvRound(x0 - 1000*(-b));
			pt2.y = cvRound(y0 - 1000*(a));

			//line( cdst, pt1, pt2, Scalar(0,0,255), 3, CV_AA);
		}
		
		
}
