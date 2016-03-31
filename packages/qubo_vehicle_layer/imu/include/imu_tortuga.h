#ifndef IMU_SIM_HEADER
#define IMU_SIM_HEADER

#include "qubo_node.h"
#include "sensor_msgs/Imu.h"
#include "tortuga/imuapi.h"

class ImuTortugaNode : QuboNode{

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
	int fd;
	sensor_msgs::Imu msg;
	ros::Subscriber subscriber;

};

#endif