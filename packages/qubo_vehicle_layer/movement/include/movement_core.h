#ifndef MOVE_CORE_H
#define MOVE_CORE_H

// ROS includes.
#include "ros/ros.h"
#include "ros/time.h"

// Custom message includes. Auto-generated from msg/ directory.
#include <sensor_msgs/Joy.h>

// Dynamic reconfigure includes.
#include <dynamic_reconfigure/server.h>
// Auto-generated from cfg/ directory.
#include <node_example/node_example_paramsConfig.h>

class moveNode
{
public:
  //! Constructor.
  moveNode();

  //! Destructor.
  ~moveNode();

  //! Callback function for dynamic reconfigure server.
  void configCallback(const sensor_msgs::Joy::ConstPtr& joy);

  //! Publish the message.
  void publishMessage(ros::Publisher *pub_message);

  //! Callback function for subscriber.
  void messageCallback(const node_example::node_example_data::ConstPtr &msg);

  //! The actual message.
  std::string message;

  //! The first integer to use in addition.
  int a;

  //! The second integer to use in addition.
  int b;
};

#endif // MOVE_CORE_H

