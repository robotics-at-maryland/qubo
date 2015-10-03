#include "test.h"

void printHello(int argc, char **argv){
  ros::init(argc, argv, "hello_world_node");
  ros::start();
  ROS_INFO_STREAM("Hello, world");
  ros::spin();
  ros::shutdown();
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "hello_world_node");
  ros::start();
  ROS_INFO_STREAM("Hello, world");
  ros::spin();
  ros::shutdown();
}
