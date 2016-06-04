#include "sensor_board_tortuga.h"


//construtor for sensor board, you need to open the sensor board before making an instance of this class. This is to allow multiple instances to talk to the same board. 
SensorBoardTortugaNode::SensorBoardTortugaNode(std::shared_ptr<ros::NodeHandle> n , int rate, int board_fd, std::string board_file): RamNode(n) {
    ros::Rate loop_rate(rate);
    fd = board_fd;
    sensor_file = board_file;
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
