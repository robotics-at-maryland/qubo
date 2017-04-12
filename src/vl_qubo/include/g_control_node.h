#ifndef G_CONTROL_NODE_H
#define G_CONTROL_NODE_H

//ros includes
#include "ros/ros.h"
#include "sensor_msgs/Imu.h"
#include "sensor_msgs/FluidPressure.h"
#include "nav_msgs/Odometry.h"
#include "std_msgs/Float64.h"

//c++ library includes
#include <iostream>
#include <sstream>
#include <thread>
#include <stdio.h>

//tf includes
#include "tf/tf.h"

//uuv includes
#include "uuv_gazebo_ros_plugins_msgs/FloatStamped.h"

#define NUM_THRUSTERS 8

class GControlNode{
    public:
    GControlNode(ros::NodeHandle n, std::string node_name, std::string pose_topic);
    ~GControlNode();

    void update();

    protected:
	
    std::string _node_name;
   
    
    //--------------------------------------------------------------------------
    //for now I'm only going to populate the orientation parameters..
       
    ros::Subscriber _orient_sub;
	void orientCallback(const sensor_msgs::Imu::ConstPtr& msg);
	
    ros::Publisher _orient_pub;
    
    sensor_msgs::Imu _fused_pose;
    
    //--------------------------------------------------------------------------
    //thruster variables

    //command subs
    ros::Subscriber _yaw_sub;
	void yawCallback(const std_msgs::Float64::ConstPtr& msg);

    ros::Subscriber _pitch_sub;
	void pitchCallback(const std_msgs::Float64::ConstPtr& msg);
	
    ros::Subscriber _roll_sub;
	void rollCallback(const std_msgs::Float64::ConstPtr& msg);
    
    //thruster vars may be able to get rid of these if we store messages..
    double _yaw_command = 0;
    double _pitch_command = 0;
    double _roll_command = 0;
    
    
    //thruster pub
    std::vector<uuv_gazebo_ros_plugins_msgs::FloatStamped> _thruster_commands;
    std::vector<ros::Publisher> _thruster_pubs;

    //--------------------------------------------------------------------------
    //depth/pressure subs
    
    ros::Subscriber _pressure_sub;
	void pressureCallback(const sensor_msgs::FluidPressure::ConstPtr& msg);
	double _pressure = 0;
	
	ros::Publisher  _depth_pub;

};

#endif
