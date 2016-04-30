#include "temp_tortuga.h"

int main(int argc, char **argv) {
    ros::init(argc, argv, "temp_node");

    TempSimNode *node1 = new TempSimNode(argc, argv, 1, "TEMP");

    /*
     * This is the general structure for using threads. The runThread function
     * has already been defined in QuboNode. Just call the thread here to run
     * a sub-node.
     */
    std::thread worker1(&TempSimNode::runThread, node1);
    worker1.join();
}
