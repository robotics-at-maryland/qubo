#include "temp_sensor_sim.h"

int main(int argc, char **argv) {
    ros::init(argc, argv, "temp_node");
    bool simulated = true;

    TempSimNode *node1 = new TempSimNode(argc, argv, 10, "HULL");
    
    while (ros::ok()) {
        node1->publish();
        node1->update();
    }
}
