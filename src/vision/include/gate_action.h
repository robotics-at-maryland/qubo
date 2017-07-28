#ifndef GATE_ACTION_H
#define GATE_ACTION_H

//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"


#include <opencv2/opencv.hpp>
#include <iostream>

class GateAction{
    public:
	
    GateAction();
	~GateAction();
	int updateAction(const cv::Mat cframe);

    protected:

    int m_kernel_size;
    int m_canny_thresh;
    int m_hough_thresh;
    int m_num_bins;
    double m_angle_thresh;
    
};

#endif
