#include "imu_tortuga.h"

ImuTortugaNode::ImuTortugaNode(int argc, char** argv, int rate){
	ros::Rate loop_rate(rate);
	publisher = n.advertise<sensor_msgs::Imu>("qubo/imu", 1000);
	fd = openIMU("/dev/ttyUSB0");
}

ImuTortugaNode::~ImuTortugaNode(){
	close(fd);
}

void ImuTortugaNode::update(){
	readIMUData(fd, data);

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


	ros::spinOnce();
}

void ImuTortugaNode::publish(){
	publisher.publish(msg);
}