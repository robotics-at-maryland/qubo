#include "vehicle_node.h"
#include "underwater_sensor_msgs/Pressure.h"

class QuboDepthSensorSim : QuboNode {

 public:
  QuboDepthSensorSim(int,char**);
  ~QuboDepthSensorSim();
  
  void subscribe();
  void publish();
  void depthCallBack(const underwater_sensor_msgs::Pressure msg);
  
 protected:

  int depth;
  ros::Subscriber subscriber;
  ros::Publisher  publisher;
 
  
};

void depthCallBack(const underwater_sensor_msgs::Pressure);

