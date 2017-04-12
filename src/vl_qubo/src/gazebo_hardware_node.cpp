#include "g_control_node.h"

#define WATER_DENSITY 1028.0 //gazebo gets this in the basic_qubo.xacro file in qubo_gazebo
#define GRAVITY 9.8 //should put both of these into the parameter server probably..

using namespace std;
using namespace ros;

GControlNode::GControlNode(ros::NodeHandle n, string node_name, string fused_pose_topic)
    :_node_name(node_name), _thruster_commands(NUM_THRUSTERS), _thruster_pubs(NUM_THRUSTERS) {

	//robot namespaces
	
	//gazebo_namespace is the namespace used by gazebo for the bot, this controller tries to
	//abstract it away

	//cont_namespace is the namespace for anything we offer up to external nodes, should probably put in header..
    string gazebo_namespace = "/basic_qubo/";
	string cont_namespace = "/qubo/"; //may merge controller and gazebo namespaces
	string qubo_namespace = "/qubo/";
	
	// topic names, channge them here if you need to
    string input_pose = qubo_namespace + "imu";
	
    string yaw_topic = cont_namespace + "yaw_command";
    string pitch_topic = cont_namespace + "pitch_command";
    string roll_topic = cont_namespace + "roll_command";

	//set up all publishers and subscribers
    _orient_sub = n.subscribe(input_pose, 1000, &GControlNode::orientCallback, this);
    _orient_pub = n.advertise<sensor_msgs::Imu>(fused_pose_topic.c_str(),1000);

    _yaw_sub = n.subscribe(yaw_topic, 1000, &GControlNode::yawCallback, this);
    _pitch_sub = n.subscribe(pitch_topic, 1000, &GControlNode::pitchCallback, this);
    _roll_sub = n.subscribe(roll_topic, 1000, &GControlNode::rollCallback, this);
	
	//register the thruster topics, we have 8
	string t_topic =  gazebo_namespace + "thrusters";
            
	for(int i = 0; i < NUM_THRUSTERS; i++){
        _thruster_commands[i].data = 0;

        t_topic = gazebo_namespace + "thrusters/";
        t_topic += to_string(i);
		t_topic += "/input";
		
        _thruster_pubs[i] = n.advertise<uuv_gazebo_ros_plugins_msgs::FloatStamped>(t_topic, 1000);
        cout << t_topic << endl;

    }
	
	
	
    //pressure sub, pressure comes in as pascals we're going to convert that to meters
	//see update for that conversion
	string pressure_topic = gazebo_namespace + "pressure";
	_pressure_sub = n.subscribe(pressure_topic, 1000, &GControlNode::pressureCallback, this);

	//this is going to be in meters
	string depth_topic = cont_namespace + "depth";
	n.advertise<sensor_msgs::FluidPressure>(depth_topic.c_str(), 1000);

				  
}

GControlNode::~GControlNode(){}



/*
Notes:
need to change incoming angle commands so the range is between -1 1

offer up thruster commands directly and add them in? 


*/


void GControlNode::update(){
    spinOnce(); //get all the callbacks
	
	//!!!! sum roll/pitch/yaw into thruster commands here.

	for(int i = 0; i < NUM_THRUSTERS; i++){
		_thruster_commands[i].data = 0;
		
	}


	//thruster layout found here https://docs.google.com/presentation/d/1mApi5nQUcGGsAsevM-5AlKPS6-FG0kfG9tn8nH2BauY/edit#slide=id.g1d529f9e65_0_3
	//yaw thruster/surge thrusters
	
	_thruster_commands[0].data += _yaw_command;
	_thruster_commands[1].data -= _yaw_command;
	_thruster_commands[2].data -= _yaw_command;
	_thruster_commands[3].data += _yaw_command;


	//pitch/roll thrusters
	_thruster_commands[4].data += ( _pitch_command + _roll_command);
	_thruster_commands[5].data += ( _pitch_command - _roll_command);
	_thruster_commands[6].data += (-_pitch_command - _roll_command);
	_thruster_commands[7].data += (-_pitch_command + _roll_command);

	for(int i = 0; i < NUM_THRUSTERS; i++){
		_thruster_pubs[i].publish(_thruster_commands[i]);
		//	cout << _thruster_commands[i].data << endl;
	}


	//sgillen@20171312-14:13 might want to move to an update function for
	//every piece of hardware but too lazy to do it rn
	//_depth = WATER_DENSITY*GRAVITY/_pressure;
	
	//TODO put this in a float64(?) message.
	//have depth update separately, clean up namespace strings (is there another way to do this with arguments to the node? is that worth it?)
	//clean up other things
	//test
	
}

void GControlNode::orientCallback(const sensor_msgs::Imu::ConstPtr &msg){
	
    // ROS_INFO("Seq: [%d]", msg->header.seq);
    // ROS_INFO("Position-> x: [%f], y: [%f], z: [%f]", msg->pose.pose.position.x,msg->pose.pose.position.y, msg->pose.pose.position.z);
    //ROS_INFO("Orientation-> x: [%f], y: [%f], z: [%f], w: [%f]", msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, msg->pose.pose.orientation.z, msg->pose.pose.orientation.w);
    //ROS_INFO("Vel-> Linear: [%f], Angular: [%f]", msg->twist.twist.linear.x,msg->twist.twist.angular.z);

    //add this to the header?

    //tf::Quaternion ahrsData(msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, msg->pose.pose.orientation.z, msg->pose.pose.orientation.w);
	
    //right now I just pass the data from simulated right through
	_orient_pub.publish(*msg);

}


//callbacks called whenever we get a new pitch/roll/yaw command, just stores the data locally
void GControlNode::yawCallback(const std_msgs::Float64::ConstPtr& msg){
    _yaw_command = msg->data;
}

void GControlNode::pitchCallback(const std_msgs::Float64::ConstPtr& msg){
    _pitch_command = msg->data;
}

void GControlNode::rollCallback(const std_msgs::Float64::ConstPtr& msg){
    _roll_command = msg->data;
}


void GControlNode::pressureCallback(const sensor_msgs::FluidPressure::ConstPtr& msg){
	_pressure = msg->fluid_pressure;
}
