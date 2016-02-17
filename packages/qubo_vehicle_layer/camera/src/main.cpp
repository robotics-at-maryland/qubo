#include "camera_sim.h"


int main(int argc, char **argv){

	ros::init(argc, argv, "camera_node");
	bool simulated = true;


	CameraSimNode *node = new CameraSimNode(argc, argv, 10);


	while(ros::ok()){
		node->update();
		node->publish();
	}
}