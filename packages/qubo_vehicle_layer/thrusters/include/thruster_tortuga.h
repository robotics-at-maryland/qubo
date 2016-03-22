

#ifndef THRUSTER_TORTUGA_H

#define THRUSTER_TORTUGA_H

#include "qubo_node.h"
#include "std_msgs/Float64MultiArray"

class ThrusterTortugaNode : QuboNode {

public:
	ThrusterTortugaNode(int, char**, int);
	~ThrusterSimNode();

	void update();
	void publish();

protected:
	std::Float64MultiArray msg;
	ros::Subscriber subscriber;

};

#endif