#ifndef G_CAMERA_NODE_H
#define G_CAMERA_NODE_H

//ros includes
#include "ros/ros.h"


//c++ stdlib includes


class GazeboCameraNode{

    public:
    GazeboCameraNode(ros::NodeHandle n);
	~GazeboCameraNode();

	void update();
};

#endif
