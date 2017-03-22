#ifndef G_CONTROL_NODE_H
#define G_CONTROL_NODE_H

//ros includes
#include "ros/ros.h"

//c++ library includes
#include <iostream>
#include <sstream>
#include <thread>
#include <stdio.h>

class GControlNode{
    public:
    GControlNode(ros::NodeHandle n, std::string node_name, std::string pose_topic);
    ~GControlNode();

    void update();

    protected:
    std::string _node_name;



    //--------------------------------------------------------------------------
    //for now I'm only going to populate the orientation parameters..
    sensor_msgs::Imu _msg;
    
    std::string pose_topic;

    ros::Publisher _orientPub;

   
};

#endif
