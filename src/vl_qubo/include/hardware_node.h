//ros includes
#include "ros/ros.h"
#include "std_msgs/Float64.h"

//c++ library includes
#include <iostream>
#include <sstream>
#include <thread>
#include <stdio.h>

#include "tf/tf.h"

//uuv includes
#include "uuv_gazebo_ros_plugins_msgs/FloatStamped.h"

#define NUM_THRUSTERS 8

class HardwareNode{
 public:
  HardwareNode(ros::NodeHandle n, std::string node_name);
  ~HardwareNode();

  void update();

 protected:
  std::string m_node_name;


  //thruster_vars
  double m_yaw_command = 0;
  double m_pitch_command = 0;
  double m_roll_command = 0;
  double m_depth_command = 0;
  double m_surge_command = 0;
  double m_sway_command = 0;
  
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

  //thrusters
    std::vector<uuv_gazebo_ros_plugins_msgs::FloatStamped> m_thruster_commands;
    std::vector<ros::Publisher> m_thruster_pubs;
};
