#include "gpu_vision_node.h"

GpuVisionNode::GpuVisionNode(std::shared_ptr<ros::NodeHandle> n, int rate, std::string feed0, std::string feed1, std::string feedb){

	this->n = n;
	cv::cuda::GpuMat test;
	int i = cv::cuda::getCudaEnabledDeviceCount();
	int e = cv::cuda::getDevice();
	if(i < 1){
		ROS_ERROR("There doesn't appear to be a GPU on this system\n");
		exit(0);
	}
	ROS_INFO("%d GPUs found", i);
}

GpuVisionNode::~GpuVisionNode(){
		//nothing here
}

void GpuVisionNode::update(){
	cv::cuda::DeviceInfo dev;
	ROS_INFO(dev.name() + "\n");
	ROS_INFO(dev.majorVersion() + ":" + dev.minorVersion() + "\n");
	ROS_INFO(dev.multiProcessorCount() + "\n");
	ros::spinOnce();
}
