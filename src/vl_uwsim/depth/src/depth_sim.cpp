#include "depth_sim.h"


DepthSimNode::DepthSimNode(std::shared_ptr<ros::NodeHandle> n, int rate) : RamNode(n){
	ros::Rate  loop_rate(rate);
	subscriber = n->subscribe("/g500/pressure", 1000, &DepthSimNode::depthCallBack,this);
	publisher = n->advertise< underwater_sensor_msgs::Pressure>("qubo/depth", 1000);
	
};

DepthSimNode::~DepthSimNode(){};

void DepthSimNode::update(){
	ros::spinOnce(); //the only thing we care about is depth here which updated whenever we get a depth call back, on a real node we may need to do something else.
	publisher.publish(msg);
}


void DepthSimNode::depthCallBack(const underwater_sensor_msgs::Pressure sim_msg)
{
	msg.pressure = sim_msg.pressure;
}

