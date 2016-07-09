#include "translational_controller.h"

TranslationalController::TranslationalController(std::shared_ptr<ros::NodeHandle> n, int inputRate) : RamNode(n) {
  thrust_pub = n->advertise<std_msgs::Int64MultiArray>("/qubo/thruster_input", inputRate);
  next_state_sub = n->subscribe<nav_msgs::Odometry>("/qubo/next_state", inputRate, &TranslationalController::nextStateCallback, this);
  current_state_sub = n->subscribe<nav_msgs::Odometry>("/qubo/current_state", inputRate, &TranslationalController::currentStateCallback, this);  
}

TranslationalController::~TranslationalController() {} 

void TranslationalController::update() {
  ros::spinOnce();
  
  std_msgs::Int64MultiArray final_thrust;
  final_thrust.layout.dim.resize(1);
  final_thrust.data.resize(6);

  final_thrust.layout.dim[0].label = "Thrusters";
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

void TranslationalController::currentStateCallback(const nav_msgs::OdometryConstPtr &current) {
  x_t = current->pose.pose.position.x;
  y_t = current->pose.pose.position.y;
  z_t = current->pose.pose.position.z;

  vx_t = current->twist.twist.linear.x;
  vy_t = current->twist.twist.linear.y;
  vz_t = current->twist.twist.linear.z;
}

void TranslationalController::nextStateCallback(const nav_msgs::OdometryConstPtr &next) {
  float x_t1 = next->pose.pose.position.x;
  float y_t1 = next->pose.pose.position.y;
  float z_t1 = next->pose.pose.position.z;

  float vx_t1 = next->twist.twist.linear.x;
  float vy_t1 = next->twist.twist.linear.y;
  float vz_t1 = next->twist.twist.linear.z;

  /*PID Controller for Each of the Values */
  float error_x = x_t1 - x_t;
  float error_y = y_t1 - y_t;
  float error_z = z_t1 - z_t;

  float error_vx = vx_t1 - vx_t;
  float error_vy = vy_t1 - vy_t;
  float error_vz = vz_t1 - vz_t;
  
  sum_error_x = sum_error_x + error_vx * dt;
  sum_error_y = sum_error_y + error_vy * dt;
  sum_error_z = sum_error_z + error_vz * dt;

  float total_error_vx = K_p * error_vx + K_i * sum_error_x + K_d * (error_vx - previous_error_x) / dt;
  float total_error_vy = K_p * error_vy + K_i * sum_error_y + K_d * (error_vy - previous_error_y) / dt;
  float total_error_vz = K_p * error_vz + K_i * sum_error_z + K_d * (error_vz - previous_error_z) / dt;

  float t1_rotation_speed = 0;
  float t2_rotation_speed = 0;
  float t3_rotation_speed = 0;
  float t4_rotation_speed = 0;
  float t5_rotation_speed = 0;
  float t6_rotation_speed = 0;

  /* Thrusters need to be changed to represent the actual vehicle*/
  /*Z-Direction*/
  thrstr_1_spd = (total_error_vz + t1_rotation_speed) / MAX_THRUSTER_SPEED;
  thrstr_2_spd = (total_error_vz + t2_rotation_speed) / MAX_THRUSTER_SPEED;

  /*X-Direction*/
  thrstr_3_spd = (total_error_vx + t3_rotation_speed) / MAX_THRUSTER_SPEED;
  thrstr_4_spd = -(total_error_vx + t4_rotation_speed) / MAX_THRUSTER_SPEED;

  /*Y-Direction*/
  thrstr_5_spd = (total_error_vy + t5_rotation_speed) / MAX_THRUSTER_SPEED;
  thrstr_6_spd = (total_error_vy + t6_rotation_speed) / MAX_THRUSTER_SPEED;
}

