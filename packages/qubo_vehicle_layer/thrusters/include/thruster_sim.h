//! This will serve as the simulated version of our thrusters


#ifndef THRUSTER_SIM_HEADER //I don't really see anybody needing to inherit this class, but better safe than sorry. 
#define THRUSTER_SIM_HEADER

#include "qubo_node.h"
#include "std_msgs/Float64MultiArray"

class ThrusterSimNode : QuboNode {

 public:
  ThrusterSimNode(int,char**,int);
  ~ThrusterSimNode();
  
  void update();
  void publish();
  void thrusterCallBack(const std_msgs::Float64MultiArray msg);
  
 protected:
  std::Float64MultiArray msg;
  ros::Subscriber subscriber;
};



#endif
