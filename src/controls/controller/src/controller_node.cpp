#include "controller_core.h"

controlNode::controlNode(std::shared_ptr<ros::NodeHandle> n, int inputRate){
  next_state_pub = n->advertise<nav_msgs::Odometry>("/qubo/next_state", inputRate);
  joystick_sub = n->subscribe<std_msgs::Float64MultiArray>("/joy_pub", inputRate, &controlNode::messageCallback, this);  
}

controlNode::~controlNode() {}

void controlNode::update() {
  ros::spinOnce();

  current_time = ros::Time::now();

  nav_msgs::Odometry next_state;
  next_state.header.stamp = current_time;
  next_state.header.frame_id = "odom";

  //set the position
  next_state.pose.pose.position.x = dx;
  next_state.pose.pose.position.y = dy;
  next_state.pose.pose.position.z = dz;
  next_state.pose.pose.orientation = next_state_orient;

  //set the velocity
  next_state.child_frame_id = "base_link";
  next_state.twist.twist.linear.x = vx;
  next_state.twist.twist.linear.y = vy;
  next_state.twist.twist.linear.z = vz;

  next_state.twist.twist.angular.y = vth;
 
  next_state_pub.publish(next_state);
}

void controlNode::messageCallback(const std_msgs::Float64MultiArray::ConstPtr &msg) {
  //current_time = ros::Time::now();
  //last_time = ros::Time::now();
  
  float a_x = msg->data[0];
  float a_y = msg->data[1];
  float a_z = msg->data[2];
  /*Twist the joystick and see what value this is*/
  float twist = msg->data[3];

  dt = 0.10; //Add this later for better accuracy: (current_time - last_time).toSec();
  
  vx = a_x * dt;
  vy = a_y * dt;
  vz = a_z * dt;

  vth = twist * dt;

  /* This only includes 2D rotations 3D will be added later and will involve quaternions.*/
  th = 0.5 * th * dt;

  dx = (vx * cos(th) - vy * sin(th)) * dt;
  dy = (vx * sin(th) + vy * cos(th)) * dt;
  dz = 0.5 * vz * dt;
  
  next_state_orient = tf::createQuaternionMsgFromYaw(th);

  ROS_DEBUG("stopped here");
}
