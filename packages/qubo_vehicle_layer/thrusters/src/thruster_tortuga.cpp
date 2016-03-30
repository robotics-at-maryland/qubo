
#include "thruster_tortuga.h"

ThrusterTortugaNode::ThrusterTortugaNode(int argc, char **argv, int rate){
	ros::Rate loop_rate(rate);
	subscriber = n.subscribe("/quob/thruster_input", 1000, &ThrusterTortugaNode::thrusterCallBack, this);
	// copied exactly from lcdshow:
	fd = openSensorBoard("/dev/ttyUSB0");
}

ThrusterTortugaNode::~ThrusterTortugaNode(){
	setSpeeds(fd, 0, 0, 0, 0, 0, 0);
	close(fd);
}

void ThrusterTortugaNode::update(){
	ros::spinOnce();
}

void ThrusterTortugaNode::publish(){
	setSpeeds(fd, msg.data[0], msg.data[1], msg.data[2], msg.data[3], msg.data[4], msg.data[5]);
}

void ThrusterTortugaNode::thrusterCallBack(const std_msgs::Float64MultiArray sim_msg){
	msg.data = sim_msg.data;
}