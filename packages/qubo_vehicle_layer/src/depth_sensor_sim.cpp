#include "depth_sensor_sim.h"

QuboDepthSensorSim::QuboDepthSensorSim(int argc, char **argv){

  ros::Rate  loop_rate(10)
; //hard coded cuz I bad
  
};

QuboDepthSensorSim::~QuboDepthSensorSim(){};


void QuboDepthSensorSim::subscribe(){
  subscriber = n.subscribe("/g500/pressure", 1000, &QuboDepthSensorSim::depthCallBack,this);
}

void QuboDepthSensorSim::publish(){
  publisher = n.advertise< underwater_sensor_msgs::Pressure>("depth", 1000);
  
  while (ros::ok()){
  
    underwater_sensor_msgs::Pressure msg;
    msg.pressure = depth;
    publisher.publish(msg);
    
    ros::spinOnce();
    
  }
}


void QuboDepthSensorSim::depthCallBack(const underwater_sensor_msgs::Pressure msg)
{
  ROS_INFO("I heard: [%f]", msg.pressure);
  this->depth = msg.pressure;
}

