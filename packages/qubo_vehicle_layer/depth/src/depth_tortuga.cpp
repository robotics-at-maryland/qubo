#include "depth_tortuga.h"


//we pass argv and argc so in the future we can pass in command line arguments.
DepthTortugaNode::DepthTortugaNode(int argc, char **argv, int rate, std::string node_name){
  ros::Rate  loop_rate(rate);
  publisher = n.advertise< underwater_sensor_msgs::Pressure>("qubo/depth", 1000);
  //  fd = openSensorBoard("/dev/sensor");
};

DepthTortugaNode::~DepthTortugaNode(){};


void DepthTortugaNode::update(){
  msg.pressure = readDepth(sensor_fd);

  //the only thing we care about is depth here which updated whenever we get a depth call back, on a real node we may need to do something else.
  publisher.publish(msg);
}

void DepthTortugaNode::depthCallBack(const underwater_sensor_msgs::Pressure sim_msg)
{
  msg.pressure = sim_msg.pressure;
}
