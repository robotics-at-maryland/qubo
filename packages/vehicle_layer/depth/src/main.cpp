#include "depth_sim.h"
#include "depth_tortuga.h"
/**
This is the main method for our depth_sensor node
**/


int main(int argc, char **argv){

  if(argc != 4){
    ROS_ERROR("The thruster node received %i arguments which is not right\n", argc);
    exit(1);
  }
  ros::init(argc, argv, "depth_sensor_node"); /** basically always needs to be called first */


  /**
     The basic idea here is to pass in a boolean from a launch script that determines if our class is 
     a real one or a simulated one. after that they behave exactly the same as far as main is concerned
     right now though we don't have a real depth sensor, hence the comments
  **/

  std::unique_ptr<QuboNode> node;
  if(strcmp(argv[1], "simulated") == 0){
    node.reset(new DepthSimNode(argc, argv, 10)); /** 10 (the rate) is completely arbitrary */
  }else if(strcmp(argv[1], "tortuga") == 0) {
    node.reset(new DepthTortugaNode(argc, argv, 10, "DEPTH"));
  }else{
    ROS_ERROR("the passed in arguments to depth node (%s) doesn't match anything that makes sense..\n", argv[1]);
    }

  while (ros::ok()){
    node->update();
  }

}
