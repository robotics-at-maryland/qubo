#include "ros/ros.h"
#include <cstdlib>
#include <cstring>

int main(int argc, char **argv)
{
  ros::init(argc, argv, "sonar_client_node");
  ros::NodeHandle n;
  std::map<std::string, std::string> header;
  header["val1"] = "";
  header["val2"] = "";
  ros::ServiceClient client = n.serviceClient<ram_msgs::sonar_data>("sonar_data",false,header);
  ram_msgs::sonar_data srv;
  strcpy(srv.request.req,"data");
  if (client.call(srv))
  {
    ROS_INFO("Calling Sonar Service\n");
        ROS_INFO("Vector: \t<%5.4f %5.4f %5.4f>\n", srv.response.vectorXYZ[0], srv.response.vectorXYZ[1], srv.response.vectorXYZ[2]);
        ROS_INFO("Status: \t0x%02x\n", srv.status);
        ROS_INFO("Range:  \t%u\n", srv.range);
        ROS_INFO("Timestamp(secs):\t%u\n", srv.timestamp);
        ROS_INFO("Sample No:\t%u\n", srv.sample_number);
  }
  else
  {
    ROS_ERROR("Failed to call sonar service");
    return 1;
  }

  return 0;
}
 

