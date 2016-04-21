
#include "dvl_tortuga.h"

DVLTortugaNode::DVLTortugaNode(int argc, char **argv, int rate){
	ros::Rate loop_rate(rate);
	publisher = n.advertise<underwater_sensor_msgs::DVL>("qubo/dvl", 1000);
}

DVLTortugaNode::~DVLTortugaNode(){}

void DVLTortugaNode::update(){
	ros::spinOnce();
}

void DVLTortugaNode::publish(){
	publisher.publish(msg);
}

void DVLTortugaNode::dvlCallBack(const underwater_sensor_msgs::DVL sim_msg){
	
}