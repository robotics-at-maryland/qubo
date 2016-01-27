//! This will serve as the simulated version of our thrusters


#ifndef THRUSTER_SIM_HEADER //I don't really see anybody needing to inherit this class, but better safe than sorry. 
#define THRUSTER_SIM_HEADER

#include "qubo_node.h"
//#include "underwater_sensor_msgs/Pressure.h"

class ThrusterSimNode : QuboNode {

 public:
  ThrusterSimNode(int,char**,int);
  ~ThrusterSimNode();
  
  void update();
  void publish();
  void thrusterCallBack(const underwater_sensor_msgs::Pressure msg);
  
 protected:
  
  // underwater_sensor_msgs::Pressure msg;
  ros::Subscriber subscriber;
  
  
};



#endif
