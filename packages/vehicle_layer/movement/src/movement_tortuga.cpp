#include "movement_core.h"

moveNode::moveNode(std::shared_ptr<ros::NodeHandle> n, int inputRate):RamNode(n) {

}

moveNode::~moveNode() {

}

void moveNode::update() {
  std_msgs::Int64MultiArray final_thrust;
  final_thrust.layout.dim.resize(1);
  final_thrust.data.resize(6);
  final_thrust.layout.dim[0].label = "Thrust";
  final_thrust.layout.dim[0].size = 6;
  final_thrust.layout.dim[0].stride = 6;
	
  final_thrust.data[0] = thrstr_1_spd;
  final_thrust.data[1] = thrstr_2_spd;
  final_thrust.data[2] = thrstr_3_spd;
  final_thrust.data[3] = thrstr_4_spd;
  final_thrust.data[4] = thrstr_5_spd;
  final_thrust.data[5] = thrstr_6_spd;

  thrust_pub.publish(final_thrust);

  ros::spinOnce();
  // ros::Duration(0.1).sleep();	
}

void moveNode::messageCallback(const nav_msgs::Odometry::ConstPtr &msg) {
  
}


void moveNode::callback(const nav_msgs::Odometry::ConstPtr& current_state, const nav_msgs::Odometry::ConstPtr& next_state) {
  
}

