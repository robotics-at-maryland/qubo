#include "JoyReader.h"

JoyReader::JoyReader(int argc, char **argv, int rate):TortugaNode() {
    //loop_rate = new ros::Rate(rate);
    subscriber = n.subscribe<sensor_msgs::Joy>("/joy", 10, &JoyReader::joyPub, this);
    publisher = n.advertise<std_msgs::Float64MultiArray>("/joy_pub", 10);
}

JoyReader::~JoyReader(){}

void JoyReader::update(){
	
  	std_msgs::Float64MultiArray msg;
	msg.layout.dim.resize(1);
	msg.data.resize(4);
        msg.layout.dim[0].label = "Input";
        msg.layout.dim[0].size = 4;
        msg.layout.dim[0].stride = 4;
        msg.data[0] = x;
        msg.data[1] = y;
        msg.data[2] = z;
        msg.data[3] = mag;
	ROS_DEBUG("sup\n");
        publisher.publish(msg);

	ros::spinOnce();
	ros::Duration(0.1).sleep();
	//loop_rate.sleep();
}

void JoyReader::joyPub(const sensor_msgs::Joy::ConstPtr &joyInput) {
	ROS_DEBUG("here\n");
	x = joyInput->axes[0]; /* Side-to-side, between -1 and +1 */
	y = joyInput->axes[1]; 	
	z = -1*joyInput->axes[5]; 
	mag = (joyInput->axes[3]+1)/2; /* Magnitude, from 0 to +1 */
}

void JoyReader::publish() {

}
