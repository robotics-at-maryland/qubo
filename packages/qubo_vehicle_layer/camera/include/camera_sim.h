// Simulated Camera
// pulls from the uwsim camera topic
// publishes a message of type sensor_msg/Image

#ifndef CAMERA_SIM_HEADER
#define CAMERA_SIM_HEADER

#include "qubo_node.h"
#include "sensor_msgs/Image.h"

class CameraSimNode : QuboNode {

public:
	CameraSimNode(int, char**, int);
	~CameraSimNode();

	void update();
	void publish();
	void cameraCallBack(const sensor_msgs::Image msg);

protected:
	sensor_msgs::Image msg;
	ros::Subscriber subscriber;

};




#endif