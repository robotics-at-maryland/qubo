#include "imu_sim.h"

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

    std::unique_ptr<RamNode> node0;

    node0.reset(new ImuSimNode(n, 10)); //10 is complete arbitrary

	// JW: because we have two IMUs, we could split this up into
	// two different class/mains if we want each IMU to do different
	// calculations.

	//There are two IMUs on tortuga, so two nodes
	

	while (ros::ok()){
		node0->update();
    }
    
}
