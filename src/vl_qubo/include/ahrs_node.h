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


//tf includes
#include "tf/tf.h"
#include "std_msgs/Float64.h"

class AHRSQuboNode{
    public:
    AHRSQuboNode(ros::NodeHandle n, std::string node_name, std::string ahrs_device);
    ~AHRSQuboNode();


    void update();

    void updateAHRS();
                         
    protected:
    
    std::string m_node_name;

    //--------------------------------------------------------------------------
    //AHRS variables
    std::string m_ahrs_device;

    AHRS m_ahrs;
    
    //defined in AHRS driver
    AHRS::AHRSData m_ahrs_data;
    
    //not sure this will actually work for what we want to publish
    //    sensor_msgs::Imu m_msg;
    std_msgs::Float64 m_roll_msg;
    std_msgs::Float64 m_pitch_msg;
    std_msgs::Float64 m_yaw_msg;
    
	//describes how many times we should try to reconnect to the device before
    //just killing the node
	const int _MAX_CONNECTION_ATTEMPTS = 10;


    //--------------------------------------------------------------------------
    //Orientation parameters (possibly fused from AHRS/IMU)
    
    //    ros::Publisher m_orientPub;
    ros::Publisher m_roll_pub;
    ros::Publisher m_pitch_pub;
    ros::Publisher m_yaw_pub;
    
};

#endif
