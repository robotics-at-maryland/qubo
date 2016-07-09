#include "rotational_controller.h"

RotationalController::RotationalController(std::shared_ptr<ros::NodeHandle> n, int inputRate) : RamNode(n) {
  thruster_pub = n->advertise<std_msgs::Int64MultiArray>("/qubo/thruster_input", inputRate);
  next_state_sub = n->subscribe<nav_msgs::Odometry>("/qubo/next_state", inputRate, &RotationalController::nextStateCallback, this);
  current_state_sub = n->subscribe<nav_msgs::Odometry>("/qubo/current_state", inputRate, &RotationalController::currentStateCallback, this); 
}

RotationalController::~RotationalController() {} 

void RotationalController::update() {
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

  thruster_pub.publish(final_thrust);	
}

/* Converts quaternion published by current state to yaw */
void RotationalController::currentStateCallback(const nav_msgs::OdometryConstPtr &current) {
  current_yaw = tf::getYaw(current->pose.pose.orientation);
}

void RotationalController::nextStateCallback(const nav_msgs::OdometryConstPtr &next) {
  /* Calculate difference between desired and current yaw. Order of the 
     subtraction may need to be changed depending on the values that
     get published to the thrusters */
  current_error_yaw = tf::getYaw(next->pose.pose.orientation) - current_yaw;

  /* Maintain a sum of all previous yaw errors for Ki term */
  integral_error_yaw += current_error_yaw * dt;

  /* Calculate a discrete derivative between the current and previous errors */
  derivative_error_yaw = (current_error_yaw - previous_error_yaw) / dt;

  /* PID controller only for yaw to generate output to thrusters */
  control_output = Kp * current_error_yaw + Ki * integral_error_yaw + Kd * derivative_error_yaw;

  /* Update previous error */
  previous_error_yaw = current_error_yaw;

  /* WORK SOME MAGIC TO SET THRUSTERS */
}

