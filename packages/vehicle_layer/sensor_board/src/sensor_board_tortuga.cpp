#include "sensor_board_tortuga.h"
#include "thrusters.h"

SensorBoardTortugaNode::SensorBoardTortugaNode(int argc, char **argv, int rate): TortugaNode() {
    ros::Rate loop_rate(rate);
    
    ROS_DEBUG("Opening sensor board");
    sensor_file = "/dev/sensor";
    fd = openSensorBoard(sensor_file.c_str()); // Function from sensorapi.h
    ROS_DEBUG("Opened sensor board with fd %d.", sensor_fd);
    checkError(syncBoard(sensor_fd)); // Function from sensorapi.h
    ROS_DEBUG("Synced with the Board");


    TortugaThrusters thrusters = new TortugaThrusters();
}

SensorBoardTortugaNode::~SensorBoardTortugaNode() {
    // Close the sensor board
    close(fd);
}

void SensorBoardTortugaNode::update() {
    thrusters->update();
    //ros::spinOnce();
} 

