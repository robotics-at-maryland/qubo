#include "imu_tortuga.h"

ImuTortugaNode::ImuTortugaNode(int argc, char** argv, int rate){
	ros::Rate loop_rate(rate);
	publisher = n.advertise<sensor_msgs::Imu>("qubo/imu", 1000);

	n.getParam(IMU_BOARD, imu_fd);
}

ImuTortugaNode::~ImuTortugaNode(){
}

void ImuTortugaNode::update(){
	if(!readIMUData(imu_fd, data)){
		ROS_ERROR("IMU Checksum Error");
	}

	msg.header.stamp = ros::Time::now();
	msg.header.seq = ++id;
	msg.header.frame_id = "0";

	msg.orientation_covariance[0] = -1;

	msg.linear_acceleration_covariance[0] = -1;

	// Our IMU returns values in G's, but we should be publishing in m/s^2
	msg.linear_acceleration.x = data->accelX * g_in_ms2;
	msg.linear_acceleration.y = data->accelY * g_in_ms2;
	msg.linear_acceleration.z = data->accelZ * g_in_ms2;

	msg.angular_velocity_covariance[0] = -1;

	msg.angular_velocity.x = data->gyroX;
	msg.angular_velocity.y = data->gyroY;
	msg.angular_velocity.z = data->gyroZ;


	//temperature data
	tempX.temperature = data->tempX;
	tempX.header.stamp =ros::Time::now();
	tempX.header.frame_id = 0;
	tempX.header.seq = id;

	tempY.temperature = data->tempY;
	tempY.header.stamp =ros::Time::now();
	tempY.header.frame_id = 0;
	tempY.header.seq = id;

	tempZ.temperature = data->tempZ;
	tempZ.header.stamp =ros::Time::now();
	tempZ.header.frame_id = 0;
	tempZ.header.seq = id;


	ros::spinOnce();
}

void ImuTortugaNode::publish(){
	publisher.publish(msg);
	publisher.publish(tempX);
	publisher.publish(tempY);
	publisher.publish(tempZ);
}