#include "test.h"

int main(int argc, char **argv)
{
  ros::init(argc, argv, "hello_world_node");
  ros::start();
  ROS_INFO_STREAM("Hello, world");
  ros::spin();
  ros::shutdown();
}
