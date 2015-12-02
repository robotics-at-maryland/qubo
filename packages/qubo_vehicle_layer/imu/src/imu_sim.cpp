#include "imu_sim.h"

ImuSimNode::ImuSimNode(int argc, char** argv, int rate){
	ros::Rate loop_rate(rate);
	subscriber = n.subscribe("/g500/imu", 1000,  &ImuSimNode::imuCallBack, this);
	publisher = n.advertise<sensor_msgs::Imu>("qubo/imu", 1000);
};

ImuSimNode::~ImuSimNode(){};

void ImuSimNode::update(){
	ros::spinOnce();
}

void ImuSimNode::publish(){
	publisher.publish(msg);
}

void ImuSimNode::imuCallBack(const sensor_msgs::Imu sim_msg){
	msg.angular_velocity = sim_msg.angular_velocity;
	//this doesn't exist, I need to find what the actual msg is
}
