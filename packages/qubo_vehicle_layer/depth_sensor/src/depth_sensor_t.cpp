#include "depth_sensor_t.h"


//we pass argv and argc so in the future we can pass in command line arguments.
DepthTortugaNode::DepthTortugaNode(int argc, char **argv, int rate){
  ros::Rate  loop_rate(rate);
  publisher = n.advertise< underwater_sensor_msgs::Pressure>("qubo/depth", 1000);
  fd = openSensorBoard("/dev/sensor");
};

DepthTortugaNode::~DepthTortugaNode(){};


void DepthTortugaNode::update(){
  depth = readDepth(fd);
  
  //the only thing we care about is depth here which updated whenever we get a depth call back, on a real node we may need to do something else.
}

void DepthTortugaNode::publish(){ //We might be able to get rid of this and always just call publisher.publish 
  publisher.publish(msg);
}


void DepthTortugaNode::depthCallBack(const underwater_sensor_msgs::Pressure sim_msg)
{
  msg.pressure = sim_msg.pressure;
}
