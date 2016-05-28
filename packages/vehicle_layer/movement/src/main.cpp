#include "movement_core.h"

int main(int argc, char **argv) {
	ros::init(argc, argv, "move_node");
	
	std::unique_ptr<QuboNode> node;
        node.reset(new moveNode(argc, argv, 10));

    while (ros::ok()){
        node->update();
        node->publish();

    }

}
