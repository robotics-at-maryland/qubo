#include "sonar_client.h"
  
SonarClientNode::SonarClientNode(std::shared_ptr<ros::NodeHandle> n, int rate, int board_fd, std::string board_file):
  SensorBoardTortugaNode(n, rate, board_fd, board_file){
/*  std::map<std::string, std::string> header;
  header["val1"] = " ";
  header["val2"] = " ";
  ros::ServiceClient client = n.serviceClient<ram_msgs::sonar_data>("sonar_data", false, header);*/
    client = n.serviceClient<ram_msgs::sonar_data>("sonar_data", false);
    srv.request.req = "data";
    if (client.call(srv))
    {
      ROS_INFO("Calling Sonar Service\n");
      ROS_INFO("Vector: \t<%5.4f %5.4f %5.4f>\n", srv.response.vectorXYZ[0], srv.response.vectorXYZ[1], srv.response.vectorXYZ[2]);
      ROS_INFO("Status: \t0x%02x\n", srv.response.status);
      ROS_INFO("Range:  \t%lu\n", srv.response.range);
      ROS_INFO("Timestamp(secs/usecs):\t%lu / %lu\n", srv.response.timestamp_sec, srv.response.timestamp_usec);
    }
    else
    {
      ROS_ERROR("Failed to call sonar service");
    }
  }
 

