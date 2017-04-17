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

class GazeboHardwareNode{
    public:
    GazeboHardwareNode(ros::NodeHandle n, std::string node_name, std::string pose_topic);
    ~GazeboHardwareNode();

    void update();

	
    protected:	
    std::string m_node_name; //I'm not totally convinced we need this..
   
    //--------------------------------------------------------------------------
    //command variables

    //command subs
    ros::Subscriber m_yaw_sub;
	void yawCallback(const std_msgs::Float64::ConstPtr& msg);

    ros::Subscriber m_pitch_sub;
	void pitchCallback(const std_msgs::Float64::ConstPtr& msg);
	
    ros::Subscriber m_roll_sub;
	void rollCallback(const std_msgs::Float64::ConstPtr& msg);

	ros::Subscriber m_depth_sub;
	void depthCallback(const std_msgs::Float64::ConstPtr& msg);

	
    //thruster vars may be able to get rid of these if we store messages..
    double m_yaw_command = 0;
    double m_pitch_command = 0;
    double m_roll_command = 0;
    double m_depth_command = 0;

	
    //thruster pubs
    std::vector<uuv_gazebo_ros_plugins_msgs::FloatStamped> m_thruster_commands;
    std::vector<ros::Publisher> m_thruster_pubs;

    //--------------------------------------------------------------------------
    //depth/pressure subs
    
    ros::Subscriber m_pressure_sub;
	void pressureCallback(const sensor_msgs::FluidPressure::ConstPtr& msg);

	std_msgs::Float64 m_depth; 
	ros::Publisher  m_depth_pub;

	    
    //--------------------------------------------------------------------------
    //for now I'm only going to populate the orientation parameters..
       
    ros::Subscriber m_orient_sub;
	void orientCallback(const sensor_msgs::Imu::ConstPtr& msg);
	
	sensor_msgs::Imu m_fused_pose;
	ros::Publisher m_orient_pub;
    
	
};

#endif
