#include "depth_tortuga.h"


//we pass argv and argc so in the future we can pass in command line arguments.
DepthTortugaNode::DepthTortugaNode(std::shared_ptr<ros::NodeHandle> n,  int rate, int board_fd, std::string board_file):
	SensorBoardTortugaNode(n, rate, board_fd, board_file){
	ros::Rate  loop_rate(rate);
	publisher = n->advertise< underwater_sensor_msgs::Pressure>("qubo/depth", 1000);
}

DepthTortugaNode::~DepthTortugaNode(){};

void DepthTortugaNode::update(){
  ros::spinOnce();
	ROS_DEBUG("READING DEPTH");

	msg.pressure = readDepth(fd);
	publisher.publish(msg);
}


//TODO
void DepthTortugaNode::depthCallBack(const underwater_sensor_msgs::Pressure sim_msg)
{
  //msg.pressure = sim_msg.pressure;
}
