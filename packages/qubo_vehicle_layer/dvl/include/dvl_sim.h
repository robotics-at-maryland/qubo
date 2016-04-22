#ifndef DVL_SIM_HEADER
#define DVL_SIM_HEADER

#include "qubo_node.h"
#include "underwater_sensor_msgs/DVL.h"

class DVLSimNode : public QuboNode {

 public:
  DVLSimNode(int,char**,int);
  ~DVLSimNode();
  
  void update();
  void publish();
  void dvlCallBack(const underwater_sensor_msgs::DVL msg);
  
 protected:

  underwater_sensor_msgs::DVL msg;
  ros::Subscriber subscriber;
 
  
};



#endif