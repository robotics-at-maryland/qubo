#include "sonar_server.h"

bool sonar(ram_msgs::sonar_data::Request &req, ram_msgs::sonar_data::Response &res){
  struct sonarData sd;
  int fd = openSensorBoard("/dev/ttyUSB0");

  if(req.req == "data"){
    getSonarData(fd, &sd);
    res.vectorXYZ[0] = sd.vectorX;
    res.vectorXYZ[1] = sd.vectorY;
    res.vectorXYZ[2] = sd.vectorZ;
    res.status = sd.status;
    res.range = sd.range;
    res.timestamp_sec = sd.timeStampSec;
    res.timestamp_usec = sd.timeStampUSec;
  }
  else{
    ROS_ERROR("Request contents wrong");
    return false;
  }
  return true;
}

SonarServerNode::SonarServerNode(std::shared_ptr<ros::NodeHandle> n, int rate, int board_fd, std::string board_file):
  
  SensorBoardTortugaNode(n, rate, board_fd, board_file){
    service = n.adverticeService("sonar_data", sonar);
    ROS_DEBUG("Ready to get sonar data.");
  }
