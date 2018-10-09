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

//service includes
#include "ram_msgs/bool_bool.h"

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

    ros::Subscriber m_surge_sub;
    void surgeCallback(const std_msgs::Float64::ConstPtr& msg);

    ros::Subscriber m_sway_sub;
    void swayCallback(const std_msgs::Float64::ConstPtr& msg);


    
	//Euler angle pubs
	ros::Publisher m_roll_pub;
	std_msgs::Float64 m_roll_msg;
	
	ros::Publisher m_pitch_pub;
	std_msgs::Float64 m_pitch_msg;
	
	ros::Publisher m_yaw_pub;
	std_msgs::Float64 m_yaw_msg;
	
	//thruster vars may be able to get rid of these if we store messages..
    double m_yaw_command = 0;
    double m_pitch_command = 0;
    double m_roll_command = 0;
    double m_depth_command = 0;
    double m_surge_command = 0;
    double m_sway_command = 0;

    //depth/pressure subs
	ros::Subscriber m_pressure_sub;
	void pressureCallback(const sensor_msgs::FluidPressure::ConstPtr& msg);

	std_msgs::Float64 m_depth; 
	ros::Publisher  m_depth_pub;

        ros::Publisher m_surge_pub;
	std_msgs::Float64 m_surge_msg;

	ros::Publisher m_sway_pub;
	std_msgs::Float64 m_sway_msg;
	
    //thruster pubs
    std::vector<uuv_gazebo_ros_plugins_msgs::FloatStamped> m_thruster_commands;
    std::vector<ros::Publisher> m_thruster_pubs;
	
	    
    //--------------------------------------------------------------------------
    //for now I'm only going to populate the orientation parameters..
       
    ros::Subscriber m_orient_sub;
    void orientCallback(const nav_msgs::Odometry::ConstPtr& msg);

    //service to switch between position and velocity parameters
    bool togglePosVel(ram_msgs::bool_bool::Request &req, ram_msgs::bool_bool::Response &res);
    bool ss_pos = false;
	
};

#endif
