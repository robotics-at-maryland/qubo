#include "gate_action.h"

#define FRAME_WIDTH 640

using namespace cv;
using namespace std;

GateAction::GateAction(){
	
	m_kernel_size = 5; //must be odd and positive
	m_canny_thresh = 12; //we may have to make this adaptive
	m_hough_thresh = 20;
	m_angle_thresh = .14;
	m_num_bins = 12; // needs to divide image width cleanly (not really though)
	
	
};


GateAction::~GateAction(){};


// we just return an int here, the vision node can handle all the action server nonsense, if anyone see's this they should convert the other stuff to this same model
int GateAction::updateAction(const Mat cframe){


//sgillen@20171021-11:10 will need to move this to it's own function I think
Mat dst, cdst;

GaussianBlur(cframe, dst, Size( m_kernel_size, m_kernel_size ), 0, 0 );

	Canny(dst, dst, m_canny_thresh, m_canny_thresh*3, 3);

	cvtColor(dst, cdst, CV_GRAY2BGR);
	vector<Vec4i> lines;

	HoughLines(dst, lines, 1, CV_PI/180, m_hough_thresh, 50, 10 );


	vector<int> xbin_count; //TODO better name

	int bin_size = FRAME_WIDTH/m_num_bins;
	
	cout << "bin size = " << bin_size << endl; 
		
	for( size_t i = 0; i < lines.size();i++) {
		float rho = lines[i][0], theta = lines[i][1];
		
		if (theta > 3.14 - m_angle_thresh && theta < 3.14 + m_angle_thresh){
			//printf("line[%lu] = %f, %f \n", i, lines[i][0], lines[i][1]);
			Point pt1, pt2;
			double a = cos(theta), b = sin(theta);
			double x0 = a*rho, y0 = b*rho;

			
			cout << "x0 =  " << x0 << "  num bins = " << m_num_bins <<  " bin = " << (int) (x0/bin_size)+1 << endl;
			int bin = (int) x0/bin_size;
			if(bin > 0){
				
				xbin_count[(int) ((x0/bin_size))]++;
				
				pt1.x = cvRound(x0 + 1000*(-b));
				pt1.y = cvRound(y0 + 1000*(a));
				pt2.x = cvRound(x0 - 1000*(-b));
				pt2.y = cvRound(y0 - 1000*(a));
														
			}
		}
		
	}
	
	int max = 0;
	int max_i = 0;
	
	for( int i = 0; i < xbin_count.size(); i++){
		if (xbin_count[i] > max ){
			max = xbin_count[i];
			max_i = i;
		}
	}
	
	int max2 = 0;
	int max2_i = 0;

	
	//the two is arbitrary and there are probably better ways to go about this
	for( int i = 0; i < xbin_count.size(); i++){
		if (xbin_count[i] > max2 && ( i > (max_i + 2)  || i < (max_i - 2 ))){
			max2 = xbin_count[i];
				max2_i = i;
		}
	}
	
	cout << "max1 - " << max_i << endl;
	cout << "max2 - " << max2_i << endl;


int offset = FRAME_WIDTH/m_num_bins - ((bin_size*max_i + bin_size/2) + (bin_size*max2_i + bin_size/2))/2;

return offset;
	
	
}
	
