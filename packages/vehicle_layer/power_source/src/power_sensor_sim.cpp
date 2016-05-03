#include "power_sensor_sim.h"

std::string PowerSimNode::currentSource;

PowerSimNode::PowerSimNode(int argc, char **argv, int inputRate, std::string name) {
    publisher = n.advertise<ram_msgs::PowerSource>("qubo/power_source/" + name, 1000);
    rate = inputRate; 
    currentSource = "";
    sourceName = name;
    enabled = DEFAULT_STATUS;
    voltage = DEFAULT_VOLTAGE;
    current = DEFAULT_CURRENT;
    life = DEFAULT_LIFE;
    /* Record the start time */
    prevTime = std::time(nullptr);

    n.setParam("qubo/current_source", currentSource);

    /*
    n.setParam("qubo/source_voltage_" + name, DEFAULT_VOLTAGE);
    n.setParam("qubo/source_current_" + name, DEFAULT_CURRENT);
    n.setParam("qubo/source_life_" + name, DEFAULT_LIFE);
    */
    
}

PowerSimNode::~PowerSimNode() {}

void PowerSimNode::update() {
    std::string specifiedSource = "";
    std::time_t currTime = std::time(nullptr);
    double deltaTime = std::difftime(currTime, prevTime);
    prevTime = currTime;

    n.param("qubo/current_source", specifiedSource, currentSource);
    
    if (sourceName.compare(currentSource) == 0) {
        enabled = true;
        voltage -= deltaTime * voltageDrainRate;
        current -= deltaTime * currentDrainRate;
    } else {
        enabled = false;
    }

    ros::spinOnce();
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
