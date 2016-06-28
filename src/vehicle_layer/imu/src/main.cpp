#include "imu_sim.h"
#include "imu_tortuga.h"

/* Main method for the imu node
 * Follows the style initally defined in the main.cpp of the thrusters
 */

// JW: should this be hardcoded?
#define IMU_0_FILE "/dev/imu"
#define IMU_1_FILE "/dev/magboom"

int main(int argc, char **argv){

	if(argc != 4){
		ROS_ERROR("The IMU node received %i arguments which is not right\n", argc);
		exit(1);
	}

	ros::init(argc, argv, "imu_node"); //basically always needs to be called first
	std::shared_ptr<ros::NodeHandle> n(new ros::NodeHandle);

	// JW: For controls, this is where it gets interesting when switching
	// between simulated and tortuga.  Currently we only have one simulated
	// IMU, but there are two on tortuga, one in the center and another
	// offset from the robot (for various reasons I'm currently forgetting)
	// I'm not sure how or *if* we need to fix this
	std::unique_ptr<RamNode> node0;
	std::unique_ptr<RamNode> node1;

	if(strcmp(argv[1], "simulated") == 0){
		node0.reset(new ImuSimNode(n, 10)); //10 is complete arbitrary
	}else if(strcmp(argv[1], "tortuga") == 0){

	// JW: because we have two IMUs, we could split this up into
	// two different class/mains if we want each IMU to do different
	// calculations.

	//There are two IMUs on tortuga, so two nodes
		node0.reset(new ImuTortugaNode(n, 10, "IMU_0", IMU_0_FILE));
	node1.reset(new ImuTortugaNode(n, 10, "MAGBOOM_IMU_1", IMU_1_FILE));
	}else{
		ROS_ERROR("the passed in arguments to IMU node (%s) doesn't match anything that makes sense...\n", argv[1]);
	}


	while (ros::ok()){
	node0->update();
	node1->update();
	}

}
