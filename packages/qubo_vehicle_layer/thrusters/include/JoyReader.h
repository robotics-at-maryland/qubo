#include "ros/ros.h"
#include <iostream>
#include "std_msgs/String.h"
#include <sstream>
#include <thread>
#include "sensor_msgs/Joy.h"


class JoyReader : public TortugaNode {

 public:
  JoyReader(int, char**, int);
  ~JoyReader();

  void update();
  void publish();
  void joyPub(const std_msgs::Float64MultiArray );

}
