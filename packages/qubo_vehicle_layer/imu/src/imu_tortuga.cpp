#include "imu_tortuga.h"

ImuTortugaNode::ImuTortugaNode(int argc, char** argv, int rate){
	ros::Rate loop_rate(rate);
	publisher = n.advertise<sensor_msgs::Imu>("qubo/imu", 1000);
	temp = n.advertise<std_msgs::Float64MultiArray>("qubo/imu/temperature", 1000);
	
	temperature.layout.data_offset = 0;
	temperature.layout.dim[0].label = "IMU Temperature";
	temperature.layout.dim[0].size = 3;
	temperature.layout.dim[0].stride = 3;
}

ImuTortugaNode::~ImuTortugaNode(){}

void ImuTortugaNode::update(){
	checkError(readIMUData(imu_fd, data));

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
	//its a float 64 array, in x, y, z order

	temperature.data[0] = data->tempX;
	temperature.data[1] = data->tempY;
	temperature.data[2] = data->tempZ;



	ros::spinOnce();
}

void ImuTortugaNode::publish(){
	publisher.publish(msg);
	temp.publish(temperature);
}

