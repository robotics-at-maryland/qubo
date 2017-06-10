#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/core/cuda.hpp"


#include "ros/ros.h"
#include <iostream>
#include <string.h>

class GpuVisionNode{
public:

	/**
	 * Everything is the same as VisionNode, just look there
	 */
	GpuVisionNode(std::shared_ptr<ros::NodeHandle> n, std::string feed0, std::string feed1, std::string feedb);
	~GpuVisionNode();
	void update();
    
    protected:
	std::shared_ptr<ros::NodeHandle> n;
};
