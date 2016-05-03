#include "sensor_board_tortuga.h"

SensorBoardTortugaNode::SensorBoardTortugaNode(int argc, char **argv, int rate): TortugaNode() {
    ros::Rate loop_rate(rate);
    
    ROS_DEBUG("Opening sensor board");
    sensor_file = "/dev/sensor";
    fd = openSensorBoard(sensor_file.c_str()); // Function from sensorapi.h
    ROS_DEBUG("Opened sensor board with fd %d.", sensor_fd);
    checkError(syncBoard(sensor_fd)); // Function from sensorapi.h
    ROS_DEBUG("Synced with the Board");
    
    if(strcmp(argv[1], "simulated") == 0) {
        
    } else if (strcmp(argv[1], "tortuga") == 0) {
        thrusters.reset(new ThrusterTortugaNode(argc, argv, 10, fd, sensor_file));
        thrusters.reset(new TempTortugaNode(argc, argv, 10, fd, sensor_file));
    } else {
        ROS_ERROR("the pased in arguments to sensor board node (%s) doesn't match anything that makes sense...", argv[1]);
        exit(1);
    }
}

SensorBoardTortugaNode::~SensorBoardTortugaNode() {
    // Close the sensor board
    close(fd);
}

void SensorBoardTortugaNode::update() {
    thrusters->update();
    ros::spinOnce();
} 

