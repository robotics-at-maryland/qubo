#include "temp_sim.h"

int main(int argc, char **argv) {
    ros::init(argc, argv, "temp_node");
    bool simulated = true;

    TempSimNode *node1 = new TempSimNode(argc, argv, 1, "HULL");
    TempSimNode *node2 = new TempSimNode(argc, argv, 10, "HULL2");
    
    /*
     * This is the general structure for using threads. The runThread function
     * has already been defined in QuboNode. Just call the thread here to run
     * a sub-node.
     */ 
    std::thread worker1(&TempSimNode::runThread, node1);
    std::thread worker2(&TempSimNode::runThread, node2);
    worker1.join();
    worker2.join();
}
