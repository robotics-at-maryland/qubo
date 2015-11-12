#include "depth_sensor_sim.h"

int main(int argc, char **argv){

  ros::init(argc, argv, "depth_sensor_node"); //basically always needs to be called first
  bool simulated = true; //We'll have to pass this one in eventually 


  //if(simulated){
  DepthSimNode *node = new DepthSimNode(argc, argv, 10); //10 (the rate) is completely arbitary
  // }
  
  while (ros::ok()){
    node->update();
    node->publish();
  }

}
