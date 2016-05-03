#include "thruster_sim.h"
#include "std_msgs/Float64MultiArray.h"

ThrusterSimNode::ThrusterSimNode(int argc, char **argv, int rate){
    ros::Rate  loop_rate(rate);
    subscriber = n.subscribe("/qubo/thrusters_input", 1000, &ThrusterSimNode::thrusterCallBack,this);
    publisher = n.advertise<std_msgs::Float64MultiArray>("/g500/thrusters_input", 1000);
  
};

ThrusterSimNode::~ThrusterSimNode(){};


/*We need to read the subscriber information and pass it as a parameter in update */  
/*Turns a R3 cartesian vector into a Float64MultiArray.*/



void ThrusterSimNode::cartesianToVelocity(double velocity []){
    /*This is the data that will be contained in the velocity vector */
    double xPower = velocity[0];
    double yPower = velocity[1];
    double zPower = velocity[2];
    
    double data [5] = {xPower/2,xPower/2,yPower,zPower/2,zPower/2};
    
    msg.layout.dim[0].label = "thrust";
    msg.layout.dim[0].size = 5;
    msg.layout.dim[0].stride = 5;
    msg.data[0] = xPower/2;
    msg.data[1] = xPower/2;
    msg.data[2] = yPower;
    msg.data[3] = zPower/2;
    msg.data[4] = zPower/2;
	
}

void ThrusterSimNode::update(){
    ros::spinOnce(); 
}

void ThrusterSimNode::publish(){ //We might be able to get rid of this and always just call publisher.publish 
    publisher.publish(msg);
}

void ThrusterSimNode::thrusterCallBack(const std_msgs::Float64MultiArray sim_msg)
{
    //cartesianToVelocity();//pass in sim_msg.data; values for cartesian velocity, and then it will be published by main/publish
}

