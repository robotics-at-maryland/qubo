#include "thruster_sim.h"
#include "std_msgs/Float64MultiArray.h"

ThrusterSimNode::ThrusterSimNode(int argc, char **argv, int rate){
  ros::Rate  loop_rate(rate);
  subscriber = n.subscribe("/qubo/thrusters_input", 1000, &ThrusterSimNode::thrusterCallBack,this);
  publisher = n.advertise<std_msgs::Float64MultiArray>("/g500/thrusters_input", 1000);
  
};

ThrusterSimNode::~ThrusterSimNode(){};


  
/*Turns a R3 cartesian vector into a Float64MultiArray.*/
void ThrusterSimNode::cartesianToVelocity(double velocity[]){

	/*This is the data that will be contained in the velocity vector */
	double xPower = velocity[0];
	double yPower = velocity[1];
	double zPower = velocity[2];
	
	double data [5] = {xPower/2,xPower/2,yPower,zPower/2,zPower/2};

	std_msgs::Float64MultiArray msg;

	msg.dim[0].label = "thrust";
	msg.dim[0].size = 5;
	msg.dim[0].stride = 5;
	msg.data[0] = xPower/2;
	msg.data[1] = xPower/2;
	msg.data[2] = yPower;
	msg.data[3] = zPower/2;
	msg.data[4] = zPower/2;
	
	publisher.publish(msg);
}

void ThrusterSimNode::update(){
  ros::spinOnce(); //the only thing we care about is depth here which updated whenever we get a depth call back, on a real node we may need to do something else.
}

void ThrusterSimNode::publish(){ //We might be able to get rid of this and always just call publisher.publish 
  publisher.publish(msg);
}

void ThrusterSimNode::thrusterCallBack(const std_msgs::Float64MultiArray sim_msg)
{
  msg.data = sim_msg.data;
}

