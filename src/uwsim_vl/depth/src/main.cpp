#include "depth_sim.h"

/**
This is the main method for our depth_sensor node
The basic idea here is to pass in a boolean from a launch script that determines if our class is 
a real one or a simulated one. after that they behave exactly the same as far as main is concerned
right now though we don't have a real depth sensor, hence the comments
**/


int main(int argc, char **argv){

	if(argc != 4){
		ROS_ERROR("The depth node received %i arguments which is not right\n", argc);
		exit(1);
	}
	
  std::shared_ptr<ros::NodeHandle> n;
  ros::init(argc, argv, "depth_sensor_node");
  
  
  std::unique_ptr<RamNode> node;


  if(strcmp(argv[1], "simulated") == 0){
	  node.reset(new DepthSimNode(n, 10)); /** 10 (the rate) is completely arbitrary */
  }else if(strcmp(argv[1], "tortuga") == 0) {
	  ROS_ERROR("The depth_tortuga  node was tried to launch independently, you want to do it through the sensor board, now exiting");
	  exit(1);
  }else{
	  ROS_ERROR("the passed in arguments to depth node (%s) doesn't match anything that makes sense..\n", argv[1]);
  }
  
  while (ros::ok()){
	  node->update();
  }

}
