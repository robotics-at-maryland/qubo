#include "movement_core.h"

/*--------------------------------------------------------------------
 * NodeExample()
 * Constructor.
 *------------------------------------------------------------------*/

moveNode::moveNode(int argc, char **argv, int inputRate, std::string name) {
	ros::Rate loop_rate(rate);
	thrust_pub = n.advertise</*thruster message*/>("file location", 1);
	joystick_sub = n.subscribe<sensor_msgs::Joy>("/joy", 1000, &moveNode::messageCallback, this);  

} // end NodeExample()

/*--------------------------------------------------------------------
 * ~NodeExample()
 * Destructor.
 *------------------------------------------------------------------*/

moveNode::~moveNode() {

} // end ~NodeExample()

/*--------------------------------------------------------------------
 * Update()
 * Publish the message.
 *------------------------------------------------------------------*/
void moveNode::update() {
	ros::spinOnce();
	
} //end update()

/*--------------------------------------------------------------------
 * publishMessage()
 * Publish the message.
 *------------------------------------------------------------------*/

void moveNode::publishMessage(ros::Publisher *pub_message) {
  node_example::node_example_data msg;
  msg.message = message;
  msg.a = a;
  msg.b = b;

  pub_message->publish(msg);
} // end publishMessage()

/*--------------------------------------------------------------------
 * messageCallback()
 * Callback function for subscriber.
 *------------------------------------------------------------------*/

void moveNode::messageCallback(const sensor_msgs::Joy::ConstPtr &msg) {
  
	message = msg->message;
	a = msg->a;
	b = msg->b;

	// Note that these are only set to INFO so they will print to a terminal for example purposes.
  	// Typically, they should be DEBUG.
  	ROS_INFO("message is %s", message.c_str());
  	ROS_INFO("sum of a + b = %d", a + b);
} // end publishCallback()

/*--------------------------------------------------------------------
 * configCallback()
 * Callback function for dynamic reconfigure server.
 *------------------------------------------------------------------*/

void moveNode::configCallback(node_example::node_example_paramsConfig &config, uint32_t level) {
  // Set class variables to new values. They should match what is input at the dynamic reconfigure GUI.
  message = config.message.c_str();
  a = config.a;
  b = config.b;
} // end configCallback()
