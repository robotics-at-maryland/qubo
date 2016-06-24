#ifndef IMU_SIM_HEADER
#define IMU_SIM_HEADER

#include "ram_node.h"
#include "sensor_msgs/Imu.h"

class ImuSimNode : public RamNode{

public:
	ImuSimNode(std::shared_ptr<ros::NodeHandle>, int);
	~ImuSimNode();

	void update();
	void imuCallBack(const sensor_msgs::Imu sim_msg);

protected:
	sensor_msgs::Imu msg;
	ros::Subscriber subscriber;
	ros::Publisher publisher;

};

#endif
