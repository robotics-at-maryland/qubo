#include "temp_tortuga.h"

TempTortugaNode::TempTortugaNode(int argc, char **argv, int rate) : TortugaNode() {
    ros::Rate loop_rate(rate);

    ROS_DEBUG("Opening sensorboard");
    sensor_file = "/dev/sensor";
    fd = openSensorBoard(sensor_file.c_str());
    ROS_DEBUG("Opened sensorboard with fd %d.", fd);
    checkError(syncBoard(fd));
    ROS_DEBUG("Synced with the Board");

    for(int i = 0; i < NUM_TEMP_SENSORS; i++) {
        publishers[i] = n.advertise<ram_msgs::Temp>("qubo/temp" + i, 1000);             
    }

} 

TempTortugaNode::TempTortugaNode(int argc, char **argv, int rate, int board_fd, std::string board_file) : TortugaNode() {
    fd = board_fd;
    sensor_file = board_file;

}

TempTortugaNode::~TempTortugaNode() {}

void TempTortugaNode::update() {
	unsigned char temps[NUM_TEMP_SENSORS];
	readTemp(fd, temps);
    for(int i = 0; i < NUM_TEMP_SENSORS; i++) {
    	msg.temp = temps[i];
        publishers[i].publish(msg);
    }

	ros::spinOnce(); //magic method always included in update
}

