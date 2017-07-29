#include "gate_action.h"

#define FRAME_WIDTH 480

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

	ROS_ERROR("1");

	//sgillen@20171021-11:10 will need to move this to it's own function I think
	Mat dst, cdst;
	
	GaussianBlur(cframe, dst, Size( m_kernel_size, m_kernel_size ), 0, 0 );

	ROS_ERROR("2");
	
	Canny(dst, dst, m_canny_thresh, m_canny_thresh*3, 3);

	ROS_ERROR("3");

	
	cvtColor(dst, cdst, CV_GRAY2BGR);
	vector<Vec2f> lines;

	ROS_ERROR("5");

	//	imshow("window" ,dst);
	//	waitKey();
	
	HoughLines(dst, lines, 1, CV_PI/180, m_hough_thresh, 50, 10 );


	
	ROS_ERROR("6");
	
	
	vector<int> xbin_count; //TODO better name

	for(int i = 0; i < m_num_bins; i++){
		xbin_count.push_back(0);
	}

	
	int bin_size = FRAME_WIDTH/m_num_bins;
	
	
	ROS_ERROR("7");
	

	cout << "bin size = " << bin_size << endl; 
		
	for( size_t i = 0; i < lines.size();i++) {
		float rho = lines[i][0], theta = lines[i][1];
		
		if (theta > 3.14 - m_angle_thresh && theta < 3.14 + m_angle_thresh){
			//printf("line[%lu] = %f, %f \n", i, lines[i][0], lines[i][1]);
			Point pt1, pt2;
			double a = cos(theta), b = sin(theta);
			double x0 = a*rho, y0 = b*rho;

			
			cout << "x0 =  " << x0 << "  num bins = " << m_num_bins <<  " bin = " << (int) (x0/bin_size) << endl;
			
			
			int bin = (int) x0/bin_size;
			
			
			if(bin > 0){
				ROS_ERROR("HERE");
				xbin_count[(int) ((x0/bin_size))]++;
				
														
			}

			ROS_ERROR("HERE@!!!ASDAS");
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


	ROS_ERROR("HERE!");
	int max2 = 0;
	int max2_i = 0;

	
	//the two is arbitrary and there are probably better ways to go about this
	for( int i = 0; i < xbin_count.size(); i++){
		if (xbin_count[i] > max2 && ( i > (max_i + 2)  || i < (max_i - 2 ))){
			max2 = xbin_count[i];
				max2_i = i;
		}
	}
	
	ROS_ERROR("AHAHAHAH");

	cout << "max1 - " << max_i << endl;
	cout << "max2 - " << max2_i << endl;


int offset = FRAME_WIDTH/m_num_bins - ((bin_size*max_i + bin_size/2) + (bin_size*max2_i + bin_size/2))/2;

return offset;
	
	
}
	
