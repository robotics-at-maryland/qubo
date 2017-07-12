#ifndef GATE_ACTION_H
#define GATE_ACTION_H

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>

class GateAction{
    public:
	
    GateAction();
	~GateAction();
	void updateAction(const cv::Mat cframe);

};

#endif
