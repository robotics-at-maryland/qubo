#include "temp_sensor_sim.h"

int main(int argc, char **argv) {
    ros::init(argc, argv, "temp_node");
    bool simulated = true;

    TempSimNode *node1 = new TempSimNode(argc, argv, 1, "HULL");
    TempSimNode *node2 = new TempSimNode(argc, argv, 10, "HULL2");
    
    std::thread worker1(&QuboNode::runThread, (QuboNode *) node1);
    std::thread worker2(&QuboNode::runThread, (QuboNode *) node2);
    worker1.join();
    worker2.join();

    /*
    while (ros::ok()) {
        node1->publish();
        node1->update();
    }*/
}
