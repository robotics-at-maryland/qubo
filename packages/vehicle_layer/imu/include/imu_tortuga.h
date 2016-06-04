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
	void imuCallBack(const sensor_msgs::Imu sim_msg);

protected:
        
        //SG: may want to make a macro.
	double g_in_ms2 = 9.80665;
	unsigned int id = 0;
	int fd = -1;
	
	//struct used to retrieve data from IMU
	std::unique_ptr<RawIMUData> data;

	//messages
	sensor_msgs::Imu msg;
	std_msgs::Float64MultiArray temperature;
	geometry_msgs::Quaternion quaternion;
	sensor_msgs::MagneticField mag;

	//publishers
	ros::Publisher imuPub;
	ros::Publisher tempPub;
	ros::Publisher quaternionPub;
	ros::Subscriber subscriberPub;
	ros::Publisher magnetsPub;

};

#endif
