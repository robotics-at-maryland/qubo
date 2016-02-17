#include "camera_sim.h"


CameraSimNode::CameraSimNode(int argc, char **argv, int rate){
	ros::Rate loop_rate(rate);
	subscriber = n.subscribe("/uwsim/camera1", 1000, &CameraSimNode::cameraCallBack, this);
	publisher = n.advertise<sensor_msgs::Image>("qubo/camera", 1000);

};


CameraSimNode::~CameraSimNode(){};

void CameraSimNode::update(){
	ros::spinOnce();
}

void CameraSimNode::publish(){
	publisher.publish(msg);
}

void CameraSimNode::cameraCallBack(const sensor_msgs::Image sim_msg){
	msg.header = sim_msg.header;
	msg.height = sim_msg.height;
	msg.width = sim_msg.width;
	msg.encoding = sim_msg.encoding;
	msg.is_bigendian = sim_msg.is_bigendian;
	msg.step = sim_msg.step;
	msg.data = sim_msg.data;
}
