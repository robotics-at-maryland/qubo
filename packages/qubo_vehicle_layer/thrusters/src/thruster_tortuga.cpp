
#include "thruster_tortuga.h"

ThrusterTortugaNode::ThrusterTortugaNode(int argc, char **argv, int rate){
	ros::Rate loop_rate(rate);
	subscriber = n.subscribe("/quob/thruster_input", 1000, &ThrusterTortugaNode::thrusterCallBack, this);
}

ThrusterTortugaNode::~ThrusterTortugaNode(){};

void ThrusterTortugaNode::update(){

}

void ThrusterTortugaNode::publish(){

}

void ThrusterTortugaNode::thrusterCallBack(const underwater_sensor_msg::Pressure sim_msg){
	
}