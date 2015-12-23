#include "power_sensor_sim.h"

int main(int argc, char **argv) {
    ros::init(argc, argv, "power_sources_node");
    bool simulated = true; 

    PowerSimNode *node1 = new PowerSimNode(argc, argv, 10, "BATT1");
    PowerSimNode *node2 = new PowerSimNode(argc, argv, 10, "BATT2");
    PowerSimNode *node3 = new PowerSimNode(argc, argv, 10, "SHORE");

    PowerSimNode::setCurrentSource("BATT1");

    while (ros::ok()) {
        node1->publish();
        node1->update();
        
        node2->publish();
        node2->update();

        node3->publish();
        node3->update();
    }
}
