#include "imu_tortuga.h"
//written by Jeremy Weed

ImuTortugaNode::ImuTortugaNode(std::shared_ptr<ros::NodeHandle> n, int rate, std::string name, std::string device) : RamNode(n){

  ros::Time::init();
	// JW: There are a lot of debug/error messages here, I'm not sure
	// if we want to leave them in or not
	ROS_DEBUG("beginning constructor");

	this->name = name;
	// JW: do I need this here?
	// SG: I think we do actually. could be completely wrong though.
	ros::Rate loop_rate(rate);

	imuPub = n->advertise<sensor_msgs::Imu>("tortuga/imu/" + name, 1000);
	tempPub = n->advertise<std_msgs::Float64MultiArray>("tortuga/imu/"+ name + "/temperature", 1000);
	quaternionPub = n->advertise<geometry_msgs::Quaternion>("tortuga/imu/" + name + "/quaternion", 1000);
	magnetsPub = n->advertise<sensor_msgs::MagneticField>("tortuga/imu/" + name + "/magnetometer", 1000);


	ROS_DEBUG("MADE IT HERE YES!");
	this->fd = openIMU(device.c_str());



	ROS_DEBUG("fd found: %d on %s", fd, name.c_str());
	if(this->fd <= 0){
            ROS_ERROR("(%s) Unable to open IMU board at: %s", name.c_str(), device.c_str());
	}




	ROS_DEBUG("end of publishers on %s", name.c_str());

	temperature.layout.dim.push_back(std_msgs::MultiArrayDimension());
	temperature.layout.data_offset = 0;
	temperature.layout.dim[0].label = "IMU Temperature";
	temperature.layout.dim[0].size = 3;
	temperature.layout.dim[0].stride = 3;
	// JW: This may fix the issues with the temp segfaults - forgot to actually reserve space for the temp on the node
	temperature.data.reserve(temperature.layout.dim[0].size);


	ROS_DEBUG("finished constructor on %s", name.c_str());
    ros::Time::init();

}

ImuTortugaNode::~ImuTortugaNode(){
	close(fd);
}

void ImuTortugaNode::update(){

	ROS_DEBUG("updating imu method on %s", name.c_str());

	static double roll = 0, pitch = 0, yaw = 0, time_last = 0;
	ROS_DEBUG("does read hang?, FD: %i", fd);
	checkError(readIMUData(this->fd, &data));
	ROS_DEBUG("nope");
	double time_current = ros::Time::now().toSec();

	msg.header.stamp = ros::Time::now();
	msg.header.seq = ++id;
	msg.header.frame_id = "odom";

	msg.orientation_covariance[0] = -1;

	msg.linear_acceleration_covariance[0] = -1;

	// Our IMU returns values in G's, but we should be publishing in m/s^2
	msg.linear_acceleration.x = data.accelX * G_IN_MS2;
	msg.linear_acceleration.y = data.accelY * G_IN_MS2;
	msg.linear_acceleration.z = data.accelZ * G_IN_MS2;

	msg.angular_velocity_covariance[0] = -1;

	msg.angular_velocity.x = data.gyroX;
	msg.angular_velocity.y = data.gyroY;
	msg.angular_velocity.z = data.gyroZ;




	//temperature data
	//its a float 64 array, in x, y, z order
	ROS_DEBUG("I BET WE DONT HAVE DATTA!\n");

//	temperature.data[0] = data.tempX;
//	temperature.data[1] = data.tempY;
//	temperature.data[2] = data.tempZ;

	ROS_DEBUG("YUP");

	//magnetometer data
	mag.header.stamp = ros::Time::now();
	mag.header.seq = id;
	mag.header.frame_id = "0";

	mag.magnetic_field.x = data.magX;
	mag.magnetic_field.y = data.magY;
	mag.magnetic_field.z = data.magZ;

	double time_delta = time_current - time_last;

/*~~~This is gross and I don't like it~~~*/

	//normalize about 2pi radians
	roll += fmod(data.gyroX / time_delta, 2 * M_PI);
	pitch += fmod(data.gyroY / time_delta, 2 * M_PI);
	yaw += fmod(data.gyroZ / time_delta, 2 * M_PI);

	//quaternion - probably
	quaternion = tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw);

    // publish data
	imuPub.publish(msg);
	tempPub.publish(temperature);
	quaternionPub.publish(quaternion);
	magnetsPub.publish(mag);

	ros::spinOnce();
}
