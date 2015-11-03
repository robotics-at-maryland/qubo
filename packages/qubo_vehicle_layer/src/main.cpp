//#include "vehicle_node.h" 
//#include "ros/ros.h"
#include "depth_sensor_sim.h"

int main(int argc, char **argv){
  
  bool simulated = true;

  // if(simulated){
    QuboDepthSensorSim *node = new QuboDepthSensorSim();
    // }
  
  
  ros::init(argc, argv, "depth_sensor_node");
  ros::NodeHandle n;
  node->subscribe();
  ros::spin();
  
}
