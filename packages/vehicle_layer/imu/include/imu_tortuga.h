#ifndef IMU_TORTUGA_HEADER
#define IMU_TORTUGA_HEADER

#include "ram_node.h"
#include "imuapi.h"

#include "sensor_msgs/Imu.h"
#include "std_msgs/Float64MultiArray.h"
#include "tf/transform_datatypes.h"
#include "geometry_msgs/Quaternion.h"
#include "sensor_msgs/MagneticField.h"

#include <cmath>


#define G_IN_MS2 9.80665



class ImuTortugaNode : public RamNode {

public:
	// JW: Currently grabs the board file from the main.cpp file
	ImuTortugaNode(std::shared_ptr<ros::NodeHandle>, int, std::string name, std::string device);
	~ImuTortugaNode();

	void update();
	void imuCallBack(const sensor_msgs::Imu sim_msg);

protected:

	unsigned int id = 0;
	int fd = -1;
	std::string name;

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

	// JW: the read method only seems to return a bool
	// telling us whether the  checksum is valid
	bool checkError(int e) {
        if(!e){
			ROS_DEBUG("IO ERROR in IMU node %s", name.c_str());
		}
    }
};

#endif
