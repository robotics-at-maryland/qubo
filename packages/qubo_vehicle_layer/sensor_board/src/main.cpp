#include "sensor_board_tortuga.h"

/**
    This is the main method for the sensor_board node
**/

int main(int argc, char **argv) {
    // Initialize sensor_board_node
    ros::init(argc, argv, "sensor_board_node");

    std::unique_ptr<QuboNode> node;

    if(strcmp(argv[1], "simulated") == 0) {
    
    } else if (strcmp(argv[1], "tortuga") == 0) {
        node.reset(new SensorBoardTortugaNode(argc, argv, 10));
    } else {
        ROS_ERROR("the pased in arguments to sensor board node (%s) doesn't match anything that makes sense...", argv[1]);
        exit(1);
    }
    
    while (ros::ok()) {
        node->update();
    }
}
