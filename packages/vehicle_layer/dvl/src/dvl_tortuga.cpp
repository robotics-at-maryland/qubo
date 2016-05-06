
#include "dvl_tortuga.h"

DVLTortugaNode::DVLTortugaNode(int argc, char **argv, int rate){
	ros::Rate loop_rate(rate);
	publisher = n.advertise<underwater_sensor_msgs::DVL>("qubo/dvl", 1000);
}

DVLTortugaNode::~DVLTortugaNode(){}

void DVLTortugaNode::update(){
    publisher.publish(msg);
	ros::spinOnce();
}

void DVLTortugaNode::dvlCallBack(const underwater_sensor_msgs::DVL sim_msg){
	
}
