#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include "ros/ros.h"
#include <iostream>

class GpuVisionNode{
public:

	/**
	 * Everything is the same as VisionNode, just look there
	 */
	GpuVisionNode(std::shared_ptr<ros::NodeHandle> n, int rate, std::string feed0, std::string feed1, std::string feedb);
	~GpuVisionNode();
	void update();

protected:
	std::shared_ptr<ros::NodeHandle> n;
};
