#include "thruster_sim.h"


ThrusterSimNode::ThrusterSimNode(int argc, char **argv, int rate){
  ros::Rate  loop_rate(rate);
  subscriber = n.subscribe("/g500/pressure", 1000, &ThrusterSimNode::depthCallBack,this);
  publisher = n.advertise< underwater_sensor_msgs::Pressure>("qubo/depth", 1000);
  
};

ThrusterSimNode::~ThrusterSimNode(){};


void ThrusterSimNode::update(){
  ros::spinOnce(); //the only thing we care about is depth here which updated whenever we get a depth call back, on a real node we may need to do something else.
}

void ThrusterSimNode::publish(){ //We might be able to get rid of this and always just call publisher.publish 
  publisher.publish(msg);
}


void ThrusterSimNode::thrusterCallBack(const underwater_sensor_msgs::Pressure sim_msg)
{
  msg.pressure = sim_msg.pressure;
}

