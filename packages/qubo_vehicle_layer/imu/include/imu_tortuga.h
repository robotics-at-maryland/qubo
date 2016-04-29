#ifndef IMU_TORTUGA_HEADER
#define IMU_TORTUGA_HEADER

#include "tortuga_node.h"
#include "sensor_msgs/Imu.h"
#include "sensor_msgs/Temperature.h"
#include "std_msgs/Float64MultiArray.h"

class ImuTortugaNode : public TortugaNode{

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
	std_msgs::Float64MultiArray temperature;
	ros::Publisher temp;
	ros::Subscriber subscriber;

};

#endif