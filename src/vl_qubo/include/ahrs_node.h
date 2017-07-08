#ifndef QUBO_AHRS_NODE_H
#define QUBO_AHRS_NODE_H

//ros includes
#include "ros/ros.h"

//c++ library includes
#include <iostream>
#include <sstream>
#include <thread>
#include <stdio.h>

//AHRS includes
#include "AHRS.h"
#include "sensor_msgs/Imu.h"


class AHRSQuboNode{
    public:
    AHRSQuboNode(ros::NodeHandle n, std::string node_name, std::string ahrs_device);
    ~AHRSQuboNode();


    void update();

    void updateAHRS();
                         
    protected:
    
    std::string _node_name;

    //--------------------------------------------------------------------------
    //AHRS variables
    std::string _ahrs_device;

    AHRS _ahrs;
    
    //defined in AHRS driver
    AHRS::AHRSData _ahrs_data;
    
    //not sure this will actually work for what we want to publish
    sensor_msgs::Imu _msg;
    
	//describes how many times we should try to reconnect to the device before
    //just killing the node
	const int _MAX_CONNECTION_ATTEMPTS = 10;


    //--------------------------------------------------------------------------
    //Orientation parameters (possibly fused from AHRS/IMU)
    
    ros::Publisher _orientPub;
    
    
};

#endif
