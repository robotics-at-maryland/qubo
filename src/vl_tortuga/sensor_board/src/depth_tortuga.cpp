#include "depth_tortuga.h"


//we pass argv and argc so in the future we can pass in command line arguments.
DepthTortugaNode::DepthTortugaNode(std::shared_ptr<ros::NodeHandle> n,  int rate, int board_fd, std::string board_file):
	SensorBoardTortugaNode(n, rate, board_fd, board_file){
	ros::Rate  loop_rate(rate);
	publisher = n->advertise< underwater_sensor_msgs::Pressure>("tortuga/depth", 1000);
}

DepthTortugaNode::~DepthTortugaNode(){};

void DepthTortugaNode::update(){
  //SG: I don't think we need this spinOnce here
  ros::spinOnce();
	ROS_DEBUG("READING DEPTH");

	msg.pressure = readDepth(fd);
	publisher.publish(msg);
}
