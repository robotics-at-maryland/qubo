#include "depth_sensor_sim.h"
#include "depth_sensor_t.h"
/**
This is the main method for our depth_sensor node
**/


int main(int argc, char **argv){

  ros::init(argc, argv, "depth_sensor_node"); /** basically always needs to be called first */
  bool simulated = true; /** We'll have to pass this one in eventually */


  /**
     The basic idea here is to pass in a boolean from a launch script that determines if our class is 
     a real one or a simulated one. after that they behave exactly the same as far as main is concerned
     right now though we don't have a real depth sensor, hence the comments
  **/


  // if(simulated){
    DepthSimNode *node = new DepthSimNode(argc, argv, 10); /** 10 (the rate) is completely arbitrary */
    //}else{
    // DepthNode *node = new DepthNode(argc, argv, 10); 
    // }


  while (ros::ok()){
    node->update();
    node->publish();
  }

}
