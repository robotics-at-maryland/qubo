#include "power_sensor_sim.h"

PowerSimNode::PowerSimNode(int argc, char **argv, int rate, std::string name) {
    ros::Rate loop_rate(rate);
    publisher = n.advertise<ram_msgs::PowerSource>("qubo/source_" + name, 1000);
}

PowerSimNode::~PowerSimNode() {}

void PowerSimNode::update() {
    // Empty since this class only publishes data.
}

void PowerSimNode::publish() {
    publisher.publish(msg);
}
