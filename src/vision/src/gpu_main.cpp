#include "gpu_vision_node.h"

int main(int argc, char** argv){
	ros::init(argc, argv,"gpu_vision_node");

	std::shared_ptr<ros::NodeHandle> n(new ros::NodeHandle);
	std::unique_ptr<GpuVisionNode> node;

	node.reset(new GpuVisionNode(n, 10, "lol0", "lol1", "lolb"));

	while(ros::ok()){
		node->update();
	}
}
