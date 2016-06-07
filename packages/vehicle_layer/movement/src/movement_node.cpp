#include "movement_core.h"

moveNode::moveNode(std::shared_ptr<ros::NodeHandle> n, int inputRate) : RamNode(n) {
  thrust_pub = n->advertise<std_msgs::Int64MultiArray>("/qubo/thruster_input", inputRate);
  next_state_sub = n->subscribe< nav_msgs::Odometry>("/qubo/next_state", inputRate, &moveNode::messageCallbackNext, this);
  current_state_sub = n->subscribe< nav_msgs::Odometry>("/qubo/current_state", inputRate, &moveNode::messageCallbackCurrent, this);  
}

moveNode::~moveNode() {} 

void moveNode::update() {
  ros::spinOnce();
  
  std_msgs::Int64MultiArray final_thrust;

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
}

void moveNode::messageCallbackCurrent(const nav_msgs::OdometryConstPtr &current) {
  x_t = current->pose.pose.position.x;
  y_t = current->pose.pose.position.y;
  z_t = current->pose.pose.position.z;

  vx_t = current->twist.twist.linear.x;
  vy_t = current->twist.twist.linear.y;
  vz_t = current->twist.twist.linear.z;
}

void moveNode::messageCallbackNext(const nav_msgs::OdometryConstPtr &next) {
  float MAX_THRUSTER_SPEED = 255;

  float x_t1 = next->pose.pose.position.x;
  float y_t1 = next->pose.pose.position.y;
  float z_t1 = next->pose.pose.position.z;

  float vx_t1 = next->twist.twist.linear.x;
  float vy_t1 = next->twist.twist.linear.y;
  float vz_t1 = next->twist.twist.linear.z;

  /*PD Controller for Each of the Values */
  float error_x = x_t1 - x_t;
  float error_y = y_t1 - y_t;
  float error_z = z_t1 - z_t;

  float error_vx = vx_t1 - vx_t;
  float error_vy = vy_t1 - vy_t;
  float error_vz = vz_t1 - vz_t;
  
  float total_error_x = K_p * error_x + K_d * error_vx * dt;
  float total_error_y = K_p * error_y + K_d * error_vy * dt;
  float total_error_z = K_p * error_z + K_d * error_vz * dt;


  /* Thrusters need to be changed to represent the actual vehicle*/
  /*Z-Direction*/
  thrstr_1_spd = total_error_z / MAX_THRUSTER_SPEED;
  thrstr_2_spd = total_error_z / MAX_THRUSTER_SPEED;

  /*X-Direction*/
  thrstr_3_spd = total_error_x / MAX_THRUSTER_SPEED;
  thrstr_4_spd = total_error_x / MAX_THRUSTER_SPEED;

  /*Y-Direction*/
  thrstr_5_spd =  total_error_y / MAX_THRUSTER_SPEED;
  thrstr_6_spd =  -1 * total_error_y / MAX_THRUSTER_SPEED;
}

