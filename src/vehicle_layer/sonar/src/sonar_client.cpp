#include "ros/ros.h"
#include <cstdlib>
#include <cstring>
#include <ram_msgs/sonar_data.h>

int main(int argc, char **argv)
{
  ros::init(argc, argv, "sonar_client_node");
  ros::NodeHandle n;
/*  std::map<std::string, std::string> header;
  header["val1"] = " ";
  header["val2"] = " ";
  ros::ServiceClient client = n.serviceClient<ram_msgs::sonar_data>("sonar_data", false, header);*/
  ros::ServiceClient client = n.serviceClient<ram_msgs::sonar_data>("sonar_data", false);
  ram_msgs::sonar_data srv;
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
    return 1;
  }

  return 0;
}
 

