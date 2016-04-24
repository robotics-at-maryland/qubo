/*This class takes joystick inputs and converts them to velocity in R3. */

JoyReader::ThrusterTortugaNode(int argc, char **argv, int rate): TortugaNode(){
    ros::Rate loop_rate(rate);
    subscriber = n.subscribe("/joy", 1000, &ThrusterTortugaNode::thrusterCallBack, this);
    publisher = n.advertise<std_msgs::Float64MultiArray>("/g500/thrusters_input", 1000); //change /g500/thrusters_input to wherever it publishes to
}

JoyReader::~ThrusterTortugaNode(){}

void JoyReader::update(){
    ros::spinOnce();
}

void JoyReader::joyPub(const std_msgs::Float64MultiArray joyInput){

	float x = joyInput.data.axes[0]; /* Side-to-side, between -1 and +1 */
	float y = joyInput.data.axes[1]; /* Forward-Backward, between -1 and +1 */
	float z = -1*joyInput.data.axes[5]; /* -1,0, or +1, defining down as positive z-values */
	float mag = (joyInput.data.axes[3]+1)/2; /* Magnitude, from 0 to +1 */

	msg.layout.dim[0].label = "Input";
	msg.layout.dim[0].size = 4;
   	msg.layout.dim[0].stride = 4;
	msg.data[0] = x;
	msg.data[1] = y;
    	msg.data[2] = z;
    	msg.data[3] = mag;
    
    publisher.publish(msg);
}

