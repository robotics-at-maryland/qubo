#include "depth_sensor_sim.h"

//QuboDepthSensorSim::QuboDepthSensorSim(){};
//QuboDepthSensorSim:: ~QuboDepthSensorSim(){};

void QuboDepthSensorSim::subscribe(){
  this->subscriber = this->sub_node.subscribe("/g500/pressure", 1000, depthCallBack);
}

void QuboDepthSensorSim::publish(){
  printf("hello!\n");
}


void depthCallBack(const underwater_sensor_msgs::Pressure msg)
{
  ROS_INFO("I heard: [%f]", msg.pressure);
}

