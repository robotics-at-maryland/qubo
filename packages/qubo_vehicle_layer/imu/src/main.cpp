#include "imu_sim.h"
#include "imu_tortuga.h"

/* Main method for the imu node
 * Follows the style initally defined in the main.cpp of the thrusters
 */


int main(int argc, char **argv){
  ROS_ERROR("begining of main");
	if(argc != 4){
		ROS_ERROR("The IMU node received %i arguments which is not right\n", argc);
		exit(1);
	}
  ros::init(argc, argv, "imu_node"); //basically always needs to be called first


  std::unique_ptr<QuboNode> node;

  if(strcmp(argv[1], "simulated") == 0){
  	node.reset(new ImuSimNode(argc, argv, 10)); //10 is complete arbitrary
  }else if(strcmp(argv[1], "tortuga") == 0){
  	node.reset(new ImuTortugaNode(argc, argv, 10));
  }else{
  	ROS_ERROR("the passed in arguments to IMU node (%s) doesn't match anything that makes sense...\n", argv[1]);
  }

  
  while (ros::ok()){
    node->update();
    node->publish();
  }

}
