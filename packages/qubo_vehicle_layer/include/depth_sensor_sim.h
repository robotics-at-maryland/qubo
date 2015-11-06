#include "vehicle_node.h"
#include "underwater_sensor_msgs/Pressure.h"

class QuboDepthSensorSim : QuboNode {

 public:
  QuboDepthSensorSim();
  ~QuboDepthSensorSim();
  
  void subscribe();
  void publish();
  
 protected:
  
  ros::NodeHandle sub_node;
  ros::Subscriber subscriber;
  
};

void depthCallBack(const underwater_sensor_msgs::Pressure);

