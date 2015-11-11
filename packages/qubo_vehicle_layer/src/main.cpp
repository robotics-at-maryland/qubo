//#include "vehicle_node.h" 
//#include "ros/ros.h"
//#include "example_device.h"
#include "depth_sensor_sim.h"

int main(int argc, char **argv){
  ros::init(argc, argv, "depth_sensor_node");
  bool simulated = true;
  
  //if(simulated){
  QuboDepthSensorSim *node = new QuboDepthSensorSim(argc, argv);
  // }
  //ExampleDevice *node = new ExampleDevice();

  node->subscribe();
  node->publish();

 
 

  ros::spin();
  
}
