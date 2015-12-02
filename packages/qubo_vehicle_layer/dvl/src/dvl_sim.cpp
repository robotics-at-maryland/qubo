#include "dvl_sim.h"

DVLSimNode::DVLSimNode(int argc, char **argv, int rate){
  ros::Rate  loop_rate(rate);
  subscriber = n.subscribe("/g500/dvl", 1000, &DVLSimNode::dvlCallBack, this);
  publisher = n.advertise<underwater_sensor_msgs::DVL>("qubo/dvl", 1000);
  
};

DVLSimNode::~DVLSimNode(){};


void DVLSimNode::update(){
  ros::spinOnce(); //the only thing we care about is depth here which updated whenever we get a depth call back, on a real node we mayneed to do something else.
}

void DVLSimNode::publish(){ //We might be able to get rid of this and always just call publisher.publish 
  publisher.publish(msg);
}


void DVLSimNode::dvlCallBack(const underwater_sensor_msgs::DVL sim_msg)
{
  msg.bi_x_axis = sim_msg.bi_x_axis;
  msg.bi_y_axis = sim_msg.bi_y_axis;
  msg.bi_z_axis = sim_msg.bi_z_axis;
}
