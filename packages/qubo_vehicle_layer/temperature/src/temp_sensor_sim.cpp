#include "temp_sensor_sim.h"

TempSimNode::TempSimNode(int argc, char **argv, int rate, std::string name) {
    ros::Rate loop_rate(rate);
    publisher = n.advertise<ram_msgs::Temperature>("qubo/temp/" + name, 1000);

    sensorName = name;
    real_temp = DEFAULT_TEMP;
    n.setParam("qubo/temp/" + sensorName, real_temp); 
} 

TempSimNode::~TempSimNode() {}

void TempSimNode::update() {
    n.param("qubo/temp/" + sensorName, real_temp, DEFAULT_TEMP);
    ros::spinOnce();
}

void TempSimNode::publish() {
    msg.temp = real_temp + unif(re);
    publisher.publish(msg);
}
