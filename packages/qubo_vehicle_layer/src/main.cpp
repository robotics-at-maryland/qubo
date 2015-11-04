//#include "vehicle_node.h" 
//#include "ros/ros.h"
#include "example_device.cpp"
//#include "depth_sensor_sim.h"

int main(int argc, char **argv){
  
  bool simulated = true;

  // if(simulated){
    //QuboDepthSensorSim *node = new QuboDepthSensorSim();
    // }
  ExampleDevice *node = new ExampleDevice();
  
  
  ros::init(argc, argv, "depth_sensor_node");
  ros::NodeHandle n;
  node->subscribe();
  ros::spin();
  
}
