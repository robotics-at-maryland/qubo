#include "gazebo_hardware_node.h"

#define WATER_DENSITY 1028.0 //gazebo gets this in the basic_qubo.xacro file in qubo_gazebo
#define GRAVITY 9.8 //should put both of these into the parameter server probably..
#define SURFACE_PRESSURE 100

using namespace std;
using namespace ros;

GazeboHardwareNode::GazeboHardwareNode(ros::NodeHandle n, string node_name, string fused_pose_topic)
    :m_node_name(node_name), m_thruster_commands(NUM_THRUSTERS), m_thruster_pubs(NUM_THRUSTERS) {

	//robot namespaces
	//TODO put these in the parameter server
	
	
	//gazebo_namespace is the namespace used by gazebo for the bot, this controller tries to
	//abstract it away

	//cont_namespace is the namespace for anything we offer up to external nodes, should probably put in header..
    string gazebo_namespace = "/basic_qubo/";
	string cont_namespace = "/qubo/"; //may merge controller and gazebo namespaces
	string qubo_namespace = "/qubo/";
	
	// topic names, channge them here if you need to
    
	
	//set up all publishers and subscribers
	string input_pose = qubo_namespace + "imu";
    m_orient_sub = n.subscribe(input_pose, 1000, &GazeboHardwareNode::orientCallback, this);
    m_orient_pub = n.advertise<sensor_msgs::Imu>(fused_pose_topic.c_str(),1000);

	
    string yaw_topic = cont_namespace + "yaw_command";
    string pitch_topic = cont_namespace + "pitch_command";
    string roll_topic = cont_namespace + "roll_command";
	string depth_command_topic = cont_namespace + "depth_command";
	
	m_yaw_sub = n.subscribe(yaw_topic, 1000, &GazeboHardwareNode::yawCallback, this);
    m_pitch_sub = n.subscribe(pitch_topic, 1000, &GazeboHardwareNode::pitchCallback, this);
    m_roll_sub = n.subscribe(roll_topic, 1000, &GazeboHardwareNode::rollCallback, this);
	m_depth_sub = n.subscribe(depth_command_topic, 1000, &GazeboHardwareNode::depthCallback, this);

	
	//register the thruster topics, we have 8
	string t_topic =  gazebo_namespace + "thrusters";
            
	for(int i = 0; i < NUM_THRUSTERS; i++){
        m_thruster_commands[i].data = 0;

        t_topic = gazebo_namespace + "thrusters/";
        t_topic += to_string(i);
		t_topic += "/input";
		
        m_thruster_pubs[i] = n.advertise<uuv_gazebo_ros_plugins_msgs::FloatStamped>(t_topic, 1000);
        cout << t_topic << endl;

    }
	
	
    //pressure sub, pressure comes in as Kilo pascals we're going to convert that to meters
	//see update for that conversion
	string pressure_topic = gazebo_namespace + "pressure";
	m_pressure_sub = n.subscribe(pressure_topic, 1000, &GazeboHardwareNode::pressureCallback, this);

	//this is going to be in meters
	string depth_topic = cont_namespace + "depth";
	m_depth_pub = n.advertise<std_msgs::Float64>(depth_topic.c_str(), 1000);

				  
}

GazeboHardwareNode::~GazeboHardwareNode(){}

/*
Notes:
need to change incoming angle commands so the range is between -1 1

offer up thruster commands directly and add them in? 


*/


void GazeboHardwareNode::update(){
    spinOnce(); //get all the callbacks, this updates all our x_command variables 

	//reset all our commands to zero
	for(int i = 0; i < NUM_THRUSTERS; i++){
		m_thruster_commands[i].data = 0;
		
	}

	
	//thruster layout found here https://docs.google.com/presentation/d/1mApi5nQUcGGsAsevM-5AlKPS6-FG0kfG9tn8nH2BauY/edit#slide=id.g1d529f9e65_0_3
	//yaw thruster/surge thrusters

	//add yaw,pitch,roll commands to our thrusters
	m_thruster_commands[0].data += m_yaw_command;
	m_thruster_commands[1].data -= m_yaw_command;
	m_thruster_commands[2].data -= m_yaw_command;
	m_thruster_commands[3].data += m_yaw_command;
	
	//pitch/roll thrusters
	m_thruster_commands[4].data += ( m_pitch_command + m_roll_command);
	m_thruster_commands[5].data += ( m_pitch_command - m_roll_command);
	m_thruster_commands[6].data += (-m_pitch_command - m_roll_command);
	m_thruster_commands[7].data += (-m_pitch_command + m_roll_command);

	
	//add in depth commands (may have to think about restructurng this?
	m_thruster_commands[4].data += m_depth_command;
	m_thruster_commands[5].data += m_depth_command;
	m_thruster_commands[6].data += m_depth_command;
	m_thruster_commands[7].data += m_depth_command;
	

	
	for(int i = 0; i < NUM_THRUSTERS; i++){
		m_thruster_pubs[i].publish(m_thruster_commands[i]);
		//	cout << _thruster_commands[i].data << endl;
	}
	
}


//------------------------------------------------------------------------------
//callbacks called whenever we get a new actuation command, just stores the data locally
//we might be more efficient by storing the message (which should stick around as long as we keep
//a reference to it.


//TODO why do we use a reference to a ConstPtr? I think we just copied some stackoveflow page or ros
//tutorial, but it doesn't make much sense to me..
void GazeboHardwareNode::yawCallback(const std_msgs::Float64::ConstPtr& msg){
    m_yaw_command = msg->data;
}

void GazeboHardwareNode::pitchCallback(const std_msgs::Float64::ConstPtr& msg){
    m_pitch_command = msg->data;
}

void GazeboHardwareNode::rollCallback(const std_msgs::Float64::ConstPtr& msg){
    m_roll_command = msg->data;
}

void GazeboHardwareNode::depthCallback(const std_msgs::Float64::ConstPtr& msg){
	m_depth_command = msg->data; 
}


//--------------------------------------------------------------------------
// Data callbacks, these take info from the simulator and pass it on in a way
// that should appear identical to how the real qubo will present it's data

void GazeboHardwareNode::orientCallback(const sensor_msgs::Imu::ConstPtr &msg){
	
    // ROS_INFO("Seq: [%d]", msg->header.seq);
    // ROS_INFO("Position-> x: [%f], y: [%f], z: [%f]", msg->pose.pose.position.x,msg->pose.pose.position.y, msg->pose.pose.position.z);
    //ROS_INFO("Orientation-> x: [%f], y: [%f], z: [%f], w: [%f]", msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, msg->pose.pose.orientation.z, msg->pose.pose.orientation.w);
    //ROS_INFO("Vel-> Linear: [%f], Angular: [%f]", msg->twist.twist.linear.x,msg->twist.twist.angular.z);
	
    //right now I just pass the data from simulated right through
	m_orient_pub.publish(*msg);

}

void GazeboHardwareNode::pressureCallback(const sensor_msgs::FluidPressure::ConstPtr& msg){
	
	//the all caps are constants defined at the beginning of this file
	//lower case are variables
	//in our case total_pressure = msg->fluid_pressure
	//fluid_pressure is in kPa, so we multiply by 1000 to get to pa and therefore a depth of meters
	
	//from - https://www.grc.nasa.gov/www/k-12/WindTunnel/Activities/fluid_pressure.html
	//total_pressure = SURFACE_PRESSURE + water_pressure
	//fluid_pressure = WATER_DENSITY*GRAVITY*depth
	//depth = total_pressure - SURFACE_PRESSURE/( WATER_DENSITY*GRAVITY)

	
	
	m_depth.data = 1000*(msg->fluid_pressure - SURFACE_PRESSURE)/(WATER_DENSITY*GRAVITY);
	m_depth_pub.publish(m_depth);
	
}
