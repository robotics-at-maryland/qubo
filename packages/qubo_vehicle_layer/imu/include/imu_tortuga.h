#ifndef IMU_TORTUGA_HEADER
#define IMU_TORTUGA_HEADER

#include "tortuga_node.h"
#include "sensor_msgs/Imu.h"
#include "std_msgs/Float64MultiArray.h"
#include "tf/transform_datatypes.h"
#include "geometry_msgs/Quaternion.h"
#include "sensor_msgs/MagneticField.h"
#include <cmath>



class ImuTortugaNode : public TortugaNode{

public:
	ImuTortugaNode(int, char**, int, std::string name, std::string device);
	~ImuTortugaNode();

	void update();
	void publish();
	void imuCallBack(const sensor_msgs::Imu sim_msg);

protected:

	double g_in_ms2 = 9.80665;
	unsigned int id = 0;
	int fd = -1;
	RawIMUData *data = NULL;
	sensor_msgs::Imu msg;
	std_msgs::Float64MultiArray temperature;
	geometry_msgs::Quaternion quaternion;
	sensor_msgs::MagneticField mag;
	ros::Publisher temp;
	ros::Publisher quaternionP;
	ros::Subscriber subscriber;
	ros::Publisher magnets;

};

#endif