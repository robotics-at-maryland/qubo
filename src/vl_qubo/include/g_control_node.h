#ifndef G_CONTROL_NODE_H
#define G_CONTROL_NODE_H

//ros includes
#include "ros/ros.h"
#include "sensor_msgs/Imu.h"
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



    void orientCallback(const sensor_msgs::Imu::ConstPtr& msg);
    void yawCallback(const std_msgs::Float64::ConstPtr& msg);
    void pitchCallback(const std_msgs::Float64::ConstPtr& msg);
    void rollCallback(const std_msgs::Float64::ConstPtr& msg);
    
    std::string _node_name;
    std::string qubo_namespace;
    
    //--------------------------------------------------------------------------
    //for now I'm only going to populate the orientation parameters..
       
    ros::Subscriber _orient_sub;
    ros::Publisher _orient_pub;
    
    sensor_msgs::Imu _fused_pose;

    
    //--------------------------------------------------------------------------
    //thruster variables

    //command subs
    ros::Subscriber _yaw_sub;
    ros::Subscriber _pitch_sub;
    ros::Subscriber _roll_sub;
    
    
    //thruster vars
    double _yaw_command = 0;
    double _pitch_command = 0;
    double _roll_command = 0;
    
    
    //thruster pub
    std::vector<uuv_gazebo_ros_plugins_msgs::FloatStamped> _thruster_commands;
    std::vector<ros::Publisher> _thruster_pubs;
    

};

#endif
