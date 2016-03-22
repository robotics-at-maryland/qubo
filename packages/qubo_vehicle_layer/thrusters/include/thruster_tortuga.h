

#ifndef THRUSTER_TORTUGA_H

#define THRUSTER_TORTUGA_H

#include "qubo_node.h"
#include "std_msgs/Float64MultiArray"
#include "sensorapi.h"

class ThrusterTortugaNode : QuboNode {

public:
	ThrusterTortugaNode(int, char**, int);
	~ThrusterSimNode();

	void update();
	void publish();
	void thrusterCallBack(const std_msgs::Float64MultiArray msg);

protected:
	std::Float64MultiArray msg;
	ros::Subscriber subscriber;

};

#endif