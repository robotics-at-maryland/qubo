#include "imu_tortuga.h"
//written by Jeremy Weed
ImuTortugaNode::ImuTortugaNode(int argc, char** argv, int rate, std::string name, std::string device){
	ROS_DEBUG("beginning constructor");

	ros::Rate loop_rate(rate);
	publisher = n.advertise<sensor_msgs::Imu>("qubo/imu/" + name, 1000);
	temp = n.advertise<std_msgs::Float64MultiArray>("qubo/imu/"+ name + "/temperature", 1000);
	quaternionP = n.advertise<geometry_msgs::Quaternion>("qubo/imu/" + name + "/quaternion", 1000);
	magnets = n.advertise<sensor_msgs::MagneticField>("qubo/imu/" + name + "/magnetometer", 1000);

	fd = openIMU(device.c_str());
	if(fd <= 0){
		ROS_ERROR("(%s) Unable to open IMU board at: %s", name.c_str(), device.c_str());
	}

	ROS_DEBUG("end of publishers");
	temperature.layout.dim.push_back(std_msgs::MultiArrayDimension());
	temperature.layout.data_offset = 0;
	temperature.layout.dim[0].label = "IMU Temperature";
	temperature.layout.dim[0].size = 3;
	temperature.layout.dim[0].stride = 3;

	ROS_DEBUG("finished constructor");
}

ImuTortugaNode::~ImuTortugaNode(){
	close(fd);
}

void ImuTortugaNode::update(){
	ROS_DEBUG("in update method");

	static double roll = 0, pitch = 0, yaw = 0, time_last = 0;
	ROS_DEBUG("does read hang?");
	checkError(readIMUData(imu_fd, data));
	ROS_DEBUG("nope");
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

	ROS_DEBUG("end of imu read");


	//temperature data
	//its a float 64 array, in x, y, z order

	temperature.data[0] = data->tempX;
	temperature.data[1] = data->tempY;
	temperature.data[2] = data->tempZ;

	//magnetometer data
	mag.header.stamp = ros::Time::now();
	mag.header.seq = id;
	mag.header.frame_id = "0";

	mag.magnetic_field.x = data->magX;
	mag.magnetic_field.y = data->magY;
	mag.magnetic_field.z = data->magZ;

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
	magnets.publish(mag);
}

