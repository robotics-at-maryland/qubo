#include "movement_core.h"


/*--------------------------------------------------------------------
 * moveNode()
 * Constructor.
 *------------------------------------------------------------------*/

moveNode::moveNode(int argc, char **argv, int inputRate):TortugaNode() {
	//ros::Rate loop_rate(rate);
	thrust_pub = n.advertise<std_msgs::Int64MultiArray>("/qubo/thruster_input", 10);
	joystick_sub = n.subscribe<std_msgs::Float64MultiArray>("/joy_pub", 10, &moveNode::messageCallback, this);  

} // end moveNode()

/*--------------------------------------------------------------------
 * ~moveNode()
 * Destructor.
 *------------------------------------------------------------------*/

moveNode::~moveNode() {

} // end ~NodeExample()

/*--------------------------------------------------------------------
 * Update()
 * Update the state of the robot.
 * Publish the message.
 *------------------------------------------------------------------*/
void moveNode::update() {
	
	std_msgs::Int64MultiArray final_thrust;
	final_thrust.layout.dim.resize(1);
	final_thrust.data.resize(6);
	final_thrust.layout.dim[0].label = "Thrust";
	final_thrust.layout.dim[0].size = 6;
	final_thrust.layout.dim[0].stride = 6;
	
  	final_thrust.data[0] = thrstr_1_spd;
  	final_thrust.data[1] = thrstr_2_spd;
	final_thrust.data[2] = thrstr_3_spd;
	final_thrust.data[3] = thrstr_4_spd;
	final_thrust.data[4] = thrstr_5_spd;
	final_thrust.data[5] = thrstr_6_spd;

  	thrust_pub.publish(final_thrust);

	ros::spinOnce();
	ros::Duration(0.1).sleep();	
} //end update()

/*--------------------------------------------------------------------
 * messageCallbacki()
 * REMEMBER CHANGE CONSTANTS 
 * Callback function for subscriber.
 *------------------------------------------------------------------*/
void moveNode::messageCallback(const std_msgs::Float64MultiArray::ConstPtr &msg) {
	float x_dir = msg->data[0];
	float y_dir = msg->data[1];
	float z_dir = msg->data[2];
	float mag = msg->data[3];
	int MAX_THRUSTER = 255; 

	if (z_dir = 0) {
		thrstr_6_spd = MAX_THRUSTER / 2;
		thrstr_4_spd = MAX_THRUSTER / 2;
				
	} else if (z_dir > 0) {
		thrstr_6_spd = MAX_THRUSTER / 3;
                thrstr_4_spd = MAX_THRUSTER / 3;	

	} else {
		thrstr_6_spd = MAX_THRUSTER * 2 / 3;
                thrstr_4_spd = MAX_THRUSTER * 2 / 3;
	}

	thrstr_3_spd = MAX_THRUSTER * x_dir / mag;
	thrstr_5_spd = -MAX_THRUSTER * x_dir / mag;

	thrstr_1_spd = MAX_THRUSTER * y_dir / mag;	
	thrstr_2_spd = MAX_THRUSTER * y_dir / mag;
} // end messageCallback

void moveNode::publish() {}
