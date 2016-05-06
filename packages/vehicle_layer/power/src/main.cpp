#include "power_sensor_sim.h"

int main(int argc, char **argv) {
    ros::init(argc, argv, "power_sources_node");
    bool simulated = true; 

    PowerSimNode *node1 = new PowerSimNode(argc, argv, 10, "BATT1");
    PowerSimNode *node2 = new PowerSimNode(argc, argv, 10, "BATT2");
    PowerSimNode *node3 = new PowerSimNode(argc, argv, 10, "SHORE");

    PowerSimNode::setCurrentSource("BATT1");

    std::thread worker1(&PowerSimNode::runThread, node1);
    std::thread worker2(&PowerSimNode::runThread, node2);
    std::thread worker3(&PowerSimNode::runThread, node3);
    
    worker1.join();
    worker2.join();
    worker3.join();

}
