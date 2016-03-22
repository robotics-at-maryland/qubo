
#include "thruster_tortuga.h"

ThrusterTortugaNode::ThrusterTortugaNode(int argc, char **argv, int rate){
	ros::Rate loop_rate(rate);
	subscriber = n.subscribe("/quob/thruster_input", 1000, &ThrusterTortugaNode::thrusterCallBack, this);
}

ThrusterTortugaNode::~ThrusterTortugaNode(){};

void ThrusterTortugaNode::update(){
	ros::spinOnce();
}

void ThrusterTortugaNode::publish(){
	//this is where we would put the implementation for intergrating 
	//the sensor api, but I'm not sure how the float array is set up
}

void ThrusterTortugaNode::thrusterCallBack(const std_msgs::Float64MultiArray sim_msg){
	msg.data =sim_msg.data;
}