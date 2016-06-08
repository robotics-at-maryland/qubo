#include "joy_reader.h"

JoyReader::JoyReader(int argc, char **argv, int rate) {
    //loop_rate = new ros::Rate(rate);
    subscriber = n.subscribe<sensor_msgs::Joy>("/joy", 10, &JoyReader::joyPub, this);
    publisher = n.advertise<std_msgs::Float64MultiArray>("/joy_pub", 10);
}

JoyReader::~JoyReader(){}

/* This sets the JoyStick reader to a standard message */
void JoyReader::update(){
	
	/* The standard message format that needs to be published */
  	std_msgs::Float64MultiArray msg;

	/* We send a 1x4 array with a vector that will be represented by three inputs */
	msg.layout.dim.resize(1);
	msg.data.resize(4);

	/* Identifying information */
	msg.layout.dim[0].label = "Input";
	msg.layout.dim[0].size = 4;
	msg.layout.dim[0].stride = 4;

	/* The message format, per Rajath's request */
	msg.data[0] = x;
	msg.data[1] = y;
	msg.data[2] = z;
	msg.data[3] = mag;

	/* Publishing messages to topic */
	publisher.publish(msg);

	ros::spinOnce();
	ros::Duration(0.1).sleep();
	//loop_rate.sleep();
}

/* Parses the data from the joystick's raw input choosing the inputs that we care about */
void JoyReader::joyPub(const sensor_msgs::Joy::ConstPtr &joyInput) {
	x = joyInput->axes[0]; /* Side-to-side, between -1 and +1 */
	y = joyInput->axes[1]; 	
	z = -1*joyInput->axes[5]; 
	mag = (joyInput->axes[3]+1)/2; /* Magnitude, from 0 to +1 */
}

