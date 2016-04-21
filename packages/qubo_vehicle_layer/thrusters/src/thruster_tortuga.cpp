#include "thruster_tortuga.h"

ThrusterTortugaNode::ThrusterTortugaNode(int argc, char **argv, int rate){
    ros::Rate loop_rate(rate);
    subscriber = n.subscribe("/qubo/thruster_input", 1000, &ThrusterTortugaNode::thrusterCallBack, this);        
}

ThrusterTortugaNode::~ThrusterTortugaNode(){
    setSpeeds(sensor_fd, 0, 0, 0, 0, 0, 0);
    //SG: does close make sense there?
}

void ThrusterTortugaNode::update(){
    //Will need more here I bet.
    ros::spinOnce();
}

void ThrusterTortugaNode::publish(){
    setSpeeds(sensor_fd, msg.data[0], msg.data[1], msg.data[2], msg.data[3], msg.data[4], msg.data[5]);
}

void ThrusterTortugaNode::thrusterCallBack(const std_msgs::Float64MultiArray sim_msg){
    //SG: TODO change this shit
    msg.data = sim_msg.data;
}
