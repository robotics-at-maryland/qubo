#include "temp_tortuga.h"

TempTortugaNode::TempTortugaNode(std::shared_ptr<ros::NodeHandle> n, int rate, int board_fd, std::string board_file):
  SensorBoardTortugaNode(n, rate, board_fd, board_file) {

  /*    ROS_DEBUG("Opening sensorboard for temperature");
    sensor_file = "/dev/sensor";
    fd = openSensorBoard(sensor_file.c_str());
    ROS_DEBUG("Opened sensorboard with fd %d.", fd);
    checkError(syncBoard(fd));
    ROS_DEBUG("Synced with the Board");
  */

    for(int i = 0; i < NUM_TEMP_SENSORS; i++) {
        publishers[i] = n->advertise<std_msgs::Char>("tortuga/temp" + std::to_string(i), 1000);
    }

} 

/* TempTortugaNode::TempTortugaNode(int argc, char **argv, int rate, int board_fd, std::string board_file) : TortugaNode() {
    fd = board_fd;
    sensor_file = board_file;
x
}
*/

TempTortugaNode::~TempTortugaNode() {}

void TempTortugaNode::update() {
  ros::spinOnce();

  unsigned char temps[NUM_TEMP_SENSORS];
  ROS_DEBUG("Reading temperatures");
  checkError(readTemp(fd, temps));
  for(int i = 0; i < NUM_TEMP_SENSORS; i++) {
    	msg.data = temps[i];
        publishers[i].publish(msg);
  }
  
}
