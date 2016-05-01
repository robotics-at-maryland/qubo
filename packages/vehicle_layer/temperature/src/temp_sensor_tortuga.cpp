#include "temp_sensor_sim.h"

TempSimNode::TempSimNode(int argc, char **argv, int inputRate, std::string name) {
    publisher = n.advertise<ram_msgs::Temperature>("qubo/temp/" + name, 1000);
    rate = inputRate;
	// don't know what the name is
 	fd = openSensorBoard("name");  
} 

TempSimNode::~TempSimNode() {}

void TempSimNode::update() {
	unsigned char temps[NUM_TEMP_SENSORS];
	readTemp(fd, temps);
	msg.data = temps;
	ros::spinOnce(); //magic method always included in update
}

void TempSimNode::publish() {	
    publisher.publish(msg);
}
