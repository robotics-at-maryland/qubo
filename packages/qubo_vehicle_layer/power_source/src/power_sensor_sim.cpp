#include "power_sensor_sim.h"

std::string PowerSimNode::currentSource;

PowerSimNode::PowerSimNode(int argc, char **argv, int rate, std::string name) {
    ros::Rate loop_rate(rate);
    publisher = n.advertise<ram_msgs::PowerSource>("qubo/source_" + name, 1000);
    
    currentSource = "";
    sourceName = name;
    enabled = DEFAULT_STATUS;
    voltage = DEFAULT_VOLTAGE;
    current = DEFAULT_CURRENT;
    life = DEFAULT_LIFE;
    /* Record the start time */
    startTime = std::time(nullptr);

    n.setParam("qubo/source_enabled_" + name, DEFAULT_STATUS);
    n.setParam("qubo/source_voltage_" + name, DEFAULT_VOLTAGE);
    n.setParam("qubo/source_current_" + name, DEFAULT_CURRENT);
    n.setParam("qubo/source_life__" + name, DEFAULT_LIFE);
    
    
}

PowerSimNode::~PowerSimNode() {}

void PowerSimNode::update() {
   std::time_t currTime = std::time(nullptr);
}

void PowerSimNode::publish() {
    msg.current_source = currentSource;
    msg.source_name = sourceName;
    msg.enabled = enabled;
    msg.voltage = voltage;
    msg.current = current;
    msg.life = life;
    publisher.publish(msg);
}

void PowerSimNode::setCurrentSource(std::string name) {
    currentSource = name; 
}
