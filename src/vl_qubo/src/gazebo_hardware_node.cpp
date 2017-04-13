#include "g_control_node.h"

#define WATER_DENSITY 1028.0 //gazebo gets this in the basic_qubo.xacro file in qubo_gazebo
#define GRAVITY 9.8 //should put both of these into the parameter server probably..
#define SURFACE_PRESSURE 100
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
	_depth_pub = n.advertise<std_msgs::Float64>(depth_topic.c_str(), 1000);

				  
}

GControlNode::~GControlNode(){}

/*
Notes:
need to change incoming angle commands so the range is between -1 1

offer up thruster commands directly and add them in? 


*/


void GControlNode::update(){
    spinOnce(); //get all the callbacks, this updates all our x_command variables 

	//reset all our commands to zero
	for(int i = 0; i < NUM_THRUSTERS; i++){
		_thruster_commands[i].data = 0;
		
	}

	
	//thruster layout found here https://docs.google.com/presentation/d/1mApi5nQUcGGsAsevM-5AlKPS6-FG0kfG9tn8nH2BauY/edit#slide=id.g1d529f9e65_0_3
	//yaw thruster/surge thrusters

	//add yaw,pitch,roll commands to our thrusters
	_thruster_commands[0].data += _yaw_command;
	_thruster_commands[1].data -= _yaw_command;
	_thruster_commands[2].data -= _yaw_command;
	_thruster_commands[3].data += _yaw_command;
	
	//pitch/roll thrusters
	_thruster_commands[4].data += ( _pitch_command + _roll_command);
	_thruster_commands[5].data += ( _pitch_command - _roll_command);
	_thruster_commands[6].data += (-_pitch_command - _roll_command);
	_thruster_commands[7].data += (-_pitch_command + _roll_command);

	
	//add in depth commands (may have to think about restructurng this?
	_thruster_commands[4].data += _depth_command;
	_thruster_commands[5].data += _depth_command;
	_thruster_commands[6].data += _depth_command;
	_thruster_commands[7].data += _depth_command;
	

	
	for(int i = 0; i < NUM_THRUSTERS; i++){
		_thruster_pubs[i].publish(_thruster_commands[i]);
		//	cout << _thruster_commands[i].data << endl;
	}


	
}


//------------------------------------------------------------------------------
//callbacks called whenever we get a new actuation command, just stores the data locally
//we might be more efficient by storing the message (which should stick around as long as we keep
//a reference to it.


//TODO why do we use a reference to a ConstPtr? I think we just copied some stackoveflow page or ros
//tutorial, but it doesn't make much sense to me..
void GControlNode::yawCallback(const std_msgs::Float64::ConstPtr& msg){
    _yaw_command = msg->data;
}

void GControlNode::pitchCallback(const std_msgs::Float64::ConstPtr& msg){
    _pitch_command = msg->data;
}

void GControlNode::rollCallback(const std_msgs::Float64::ConstPtr& msg){
    _roll_command = msg->data;
}

void GControlNode::depthCallBack(const std_msgs::Float64::ConstPtr& msg){
	_depth_command = msg->data; 
}


//--------------------------------------------------------------------------
// Data callbacks, these take info from the simulator and pass it on in a way
// that should appear identical to how the real qubo will present it's data

void GControlNode::orientCallback(const sensor_msgs::Imu::ConstPtr &msg){
	
    // ROS_INFO("Seq: [%d]", msg->header.seq);
    // ROS_INFO("Position-> x: [%f], y: [%f], z: [%f]", msg->pose.pose.position.x,msg->pose.pose.position.y, msg->pose.pose.position.z);
    //ROS_INFO("Orientation-> x: [%f], y: [%f], z: [%f], w: [%f]", msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, msg->pose.pose.orientation.z, msg->pose.pose.orientation.w);
    //ROS_INFO("Vel-> Linear: [%f], Angular: [%f]", msg->twist.twist.linear.x,msg->twist.twist.angular.z);
	
    //right now I just pass the data from simulated right through
	_orient_pub.publish(*msg);

}

void GControlNode::pressureCallback(const sensor_msgs::FluidPressure::ConstPtr& msg){
	
	//the all caps are constants defined at the beginning of this file
	//lower case are variables
	//in our case total_pressure = msg->fluid_pressure
	//fluid_pressure is in kPa, so we multiply by 1000 to get to pa and therefore a depth of meters
	
	//from - https://www.grc.nasa.gov/www/k-12/WindTunnel/Activities/fluid_pressure.html
	//total_pressure = SURFACE_PRESSURE + water_pressure
	//fluid_pressure = WATER_DENSITY*GRAVITY*depth
	//depth = total_pressure - SURFACE_PRESSURE/( WATER_DENSITY*GRAVITY)

	
	
	_depth.data = 1000*(msg->fluid_pressure - SURFACE_PRESSURE)/(WATER_DENSITY*GRAVITY);
	_depth_pub.publish(_depth);

	//have depth update separately, clean up namespace strings (is there another way to do this with arguments to the node? is that worth it?)
	//clean up other things
	//test
	
}
