//! This is the simulated version of our depth sensor

/** This node subscribes to /g500/pressure (though that will need to change when we have our own robot sim up and running) 
 *  the "pressure" is actually just the depth plus some noise, so we're using it unadulterated. 
 **/


#ifndef DEPTHSIM_HEADER //I don't really see anybody needing to inherit this class, but better safe than sorry. 
#define DEPTHSIM_HEADER

#include "qubo_node.h"
#include "underwater_sensor_msgs/Pressure.h"

class DepthSimNode : QuboNode {

 public:
  DepthSimNode(int,char**,int);
  ~DepthSimNode();
  
  void update();
  void publish();
  void depthCallBack(const underwater_sensor_msgs::Pressure msg);
  
 protected:

  underwater_sensor_msgs::Pressure msg;
  ros::Subscriber subscriber;
 
  
};



#endif
