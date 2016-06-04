#include "sensor_board_tortuga.h"


SensorBoardTortugaNode::SensorBoardTortugaNode(std::shared_ptr<ros::NodeHandle> n , int rate, int board_fd, std::string board_file): RamNode(n) {
    ros::Rate loop_rate(rate);
    fd = board_fd;
    sensor_file = board_file;
    

    /*
    ROS_DEBUG("Opening sensor board");
    sensor_file = "/dev/sensor";
    fd = openSensorBoard(sensor_file.c_str()); // Function from sensorapi.h
    ROS_DEBUG("Opened sensor board with fd %d.", sensor_fd);
    checkError(syncBoard(sensor_fd)); 
    ROS_DEBUG("Synced with the Board");
    */
}

SensorBoardTortugaNode::~SensorBoardTortugaNode() {
    // Close the sensor board
    // close(fd);
}

void SensorBoardTortugaNode::update() {
    //ros::spinOnce();
} 

//and quit/shutdown.
bool SensorBoardTortugaNode::checkError(int e) {
    switch(e) {
    case SB_IOERROR:
        ROS_DEBUG("IO ERROR in node %s", sensor_file.c_str());
        return true;
    case SB_BADCC:
        ROS_DEBUG("BAD CC ERROR in node %s", sensor_file.c_str());
        return true;
    case SB_HWFAIL:
        ROS_DEBUG("HW FAILURE ERROR in node %s", sensor_file.c_str());
        return true;
    case SB_ERROR:
        ROS_DEBUG("SB ERROR in node %s", sensor_file.c_str());
    }
}
