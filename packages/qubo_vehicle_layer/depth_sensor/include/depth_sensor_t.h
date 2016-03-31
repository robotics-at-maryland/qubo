//! This is tortugas version of the depth sensor node. 




#ifndef DEPTHT_HEADER //I don't really see anybody needing to inherit this class, but better safe than sorry. 
#define DEPTHT_HEADER

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
 
  
};



#endif
