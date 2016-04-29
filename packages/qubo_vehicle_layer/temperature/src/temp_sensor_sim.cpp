#include "temp_sensor_sim.h"

TempSimNode::TempSimNode(int argc, char **argv, int inputRate, std::string name) {
    publisher = n.advertise<ram_msgs::Temperature>("qubo/temp/" + name, 1000);
    rate = inputRate;
    sensorName = name;
    real_temp = DEFAULT_TEMP;
    n.setParam("qubo/temp/" + sensorName, real_temp);
} 

TempSimNode::~TempSimNode() {}

void TempSimNode::update() {
    double curr_temp;
    real_temp = curr_temp;
    ros::spinOnce(); //magic method always included in update
}

void TempSimNode::publish() {
    std::uniform_real_distribution<double> unif(-1.0, 1.0);
    publisher.publish(msg);
}
