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
	ROS_INFO("%s",dev.name());
	ROS_INFO("\n");
	ROS_INFO("%d",dev.majorVersion());
	ROS_INFO(":");
	ROS_INFO("%d",dev.minorVersion());
	ROS_INFO("\n");
	ROS_INFO("%d",dev.multiProcessorCount());
	ROS_INFO("\n");
	ros::spinOnce();
}
