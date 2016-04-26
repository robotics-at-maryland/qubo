#include "imu_tortuga.h"

ImuTortugaNode::ImuTortugaNode(int argc, char** argv, int rate){
	ROS_ERROR("begining constructor");

	ros::Rate loop_rate(rate);
	publisher = n.advertise<sensor_msgs::Imu>("qubo/imu", 1000);
	temp = n.advertise<std_msgs::Float64MultiArray>("qubo/imu/temperature", 1000);
	quaternionP = n.advertise<geometry_msgs::Quaternion>("qubo/imu/quaternion", 1000);
	
	ROS_ERROR("end of publishers");
	temperature.layout.dim.push_back(std_msgs::MultiArrayDimension());
	temperature.layout.data_offset = 0;
	temperature.layout.dim[0].label = "IMU Temperature";
	temperature.layout.dim[0].size = 3;
	temperature.layout.dim[0].stride = 3;

	ROS_ERROR("finished constructor");
}

ImuTortugaNode::~ImuTortugaNode(){}

void ImuTortugaNode::update(){
	ROS_ERROR("in update method");

	static double roll = 0, pitch = 0, yaw = 0, time_last = 0;
	ROS_ERROR("does read hang?");
	checkError(readIMUData(imu_fd, data));
	ROS_ERROR("nope");
	double time_current = ros::Time::now().toSec();

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

	ROS_ERROR("end of imu read");


	//temperature data
	//its a float 64 array, in x, y, z order

	temperature.data[0] = data->tempX;
	temperature.data[1] = data->tempY;
	temperature.data[2] = data->tempZ;

	
	double time_delta = time_current - time_last;

/*~~~This is gross and I don't like it~~~*/

	//normalize about 2pi radians
	roll += fmod(data->gyroX / time_delta, 2 * M_PI);
	pitch += fmod(data->gyroY / time_delta, 2 * M_PI);
	yaw += fmod(data->gyroZ / time_delta, 2 * M_PI);

	//quaternion - probably 
	quaternion = tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw);


	ros::spinOnce();
}

void ImuTortugaNode::publish(){
	publisher.publish(msg);
	temp.publish(temperature);
	quaternionP.publish(quaternion);
}

