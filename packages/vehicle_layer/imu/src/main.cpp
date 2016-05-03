#include "imu_sim.h"
#include "imu_tortuga.h"

/* Main method for the imu node
 * Follows the style initally defined in the main.cpp of the thrusters
 */
#define IMU_0_FILE "/dev/imu"
#define IMU_1_FILE "/dev/magboom"

int main(int argc, char **argv){
	if(argc != 4){
		ROS_ERROR("The IMU node received %i arguments which is not right\n", argc);
		exit(1);
	}
  ros::init(argc, argv, "imu_node"); //basically always needs to be called first


  std::unique_ptr<QuboNode> node0;
  std::unique_ptr<QuboNode> node1;

  if(strcmp(argv[1], "simulated") == 0){
  	node0.reset(new ImuSimNode(argc, argv, 10)); //10 is complete arbitrary
  }else if(strcmp(argv[1], "tortuga") == 0){
    //There are two IMUs on tortuga, so two nodes
  	node0.reset(new ImuTortugaNode(argc, argv, 10, "IMU_0", IMU_0_FILE));
    node1.reset(new ImuTortugaNode(argc, argv, 10, "IMU_1", IMU_1_FILE));
  }else{
  	ROS_ERROR("the passed in arguments to IMU node (%s) doesn't match anything that makes sense...\n", argv[1]);
  }

  
  while (ros::ok()){
    node0->update();
    node1->update();
  }

}
