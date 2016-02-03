#include "temp_sensor_sim.h"

TempSimNode::TempSimNode(int argc, char **argv, int rate, std::string name) {
    ros::Rate loop_rate(rate);
    publisher = n.advertise<ram_msgs::Temperature>("qubo/temp/" + name, 1000);

    sensorName = name;
    temp = DEFAULT_TEMP;
    n.setParam("qubo/temp/" + sensorName, temp); 
} 

TempSimNode::~TempSimNode() {}

void TempSimNode::update() {
    n.param("qubo/temp/" + sensorName, temp, DEFAULT_TEMP);
    std::uniform_real_distribution<double> unif(-1.0, 1.0);
    std::default_random_engine re;
    ros::spinOnce();
}

void TempSimNode::publish() {
}
