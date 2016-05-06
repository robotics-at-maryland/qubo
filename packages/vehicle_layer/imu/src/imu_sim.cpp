#include "imu_sim.h"

ImuSimNode::ImuSimNode(int argc, char** argv, int rate){
	ros::Rate loop_rate(rate);
	subscriber = n.subscribe("/g500/imu", 1000,  &ImuSimNode::imuCallBack, this);
	publisher = n.advertise<sensor_msgs::Imu>("qubo/imu", 1000);
};

ImuSimNode::~ImuSimNode(){};

void ImuSimNode::update(){
	ros::spinOnce();
	publisher.publish(msg);
}


void ImuSimNode::imuCallBack(const sensor_msgs::Imu sim_msg){
	msg.header = sim_msg.header;
	msg.orientation = sim_msg.orientation;
	msg.orientation_covariance = sim_msg.orientation_covariance;
	msg.angular_velocity = sim_msg.angular_velocity;
	msg.linear_acceleration = sim_msg.linear_acceleration;
	msg.linear_acceleration_covariance = sim_msg.linear_acceleration_covariance;
	//this doesn't exist, I need to find what the actual msg is
}
