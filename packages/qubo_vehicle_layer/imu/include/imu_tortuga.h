#ifndef IMU_SIM_HEADER
#define IMU_SIM_HEADER

#include "tortuga_node.h"
#include "sensor_msgs/Imu.h"
#include "sensor_msgs/Temperature.h"

class ImuTortugaNode : TortugaNode{

public:
	ImuTortugaNode(int, char**, int);
	~ImuTortugaNode();

	void update();
	void publish();
	void imuCallBack(const sensor_msgs::Imu sim_msg);

protected:

	double g_in_ms2 = 9.80665;
	unsigned int id = 0;
	RawIMUData *data = NULL;
	sensor_msgs::Imu msg;
	sensor_msgs::Temperature tempX, tempY, tempZ;
	ros::Publisher tempPub;
	ros::Subscriber subscriber;

};

#endif