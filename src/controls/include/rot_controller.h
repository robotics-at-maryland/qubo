#include "pid_controller.h"

#ifndef ROT_CONTROLLER_H
#define ROT_CONTROLLER_H


class RotController : PIDController{

	public:
	RotController(ros::NodeHandle n, std::string control_topic);
	~RotController();
	
	void update();
};

#endif
