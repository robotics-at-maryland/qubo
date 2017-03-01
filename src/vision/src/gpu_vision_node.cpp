#include "gpu_vision_node.h"

GpuVisionNode::GpuVisionNode(std::shared_ptr<ros::NodeHandle> n, std::string feed0, std::string feed1, std::string feedb){

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
	ROS_INFO("Name: %s",dev.name());
	ROS_INFO("\n");
	ROS_INFO("Compatible: %s", dev.isCompatible());
	ROS_INFO("Major version: %d",dev.majorVersion());
	ROS_INFO(":");
	ROS_INFO("Minor version: %d",dev.minorVersion());
	ROS_INFO("\n");
	ROS_INFO("multiProcessor count: %d",dev.multiProcessorCount());
	ROS_INFO("\n");
	ros::spinOnce();
}
