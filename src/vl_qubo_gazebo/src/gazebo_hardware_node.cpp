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
    string gazebo_namespace = "/qubo_gazebo/";
	string cont_namespace = "/qubo/"; //may merge controller and gazebo namespaces
	string qubo_namespace = "/qubo/";
	
	//topic names, change them here if you need to
    
	//set up all publishers and subscribers
	string input_pose = gazebo_namespace + "pose_gt";
    m_orient_sub = n.subscribe(input_pose, 1000, &GazeboHardwareNode::orientCallback, this);

	//pressure sub, pressure comes in as Kilo pascals we're going to convert that to meters
	//giving us the data we publish over the depth topic. see update for that conversion
	string pressure_topic = gazebo_namespace + "pressure";
	m_pressure_sub = n.subscribe(pressure_topic, 1000, &GazeboHardwareNode::pressureCallback, this);
	
	
	//roll pitch yaw + depth topics. We output our current best guess at these parameters using the
	//topic strings below, we subscribe to commands for each DOF on the <axis>_cmd topics. 
	string roll_topic = cont_namespace + "roll";
    string pitch_topic = cont_namespace + "pitch";
	string yaw_topic = cont_namespace + "yaw";
	string depth_topic = cont_namespace + "depth"; //sometimes called heave 
	string surge_topic = cont_namespace + "surge"; //"forward" translational motion 
	string sway_topic = cont_namespace + "sway";   //"sideways" translational motion

	m_roll_sub  = n.subscribe(roll_topic  + "_cmd", 1000, &GazeboHardwareNode::rollCallback, this);
	m_pitch_sub = n.subscribe(pitch_topic + "_cmd", 1000, &GazeboHardwareNode::pitchCallback, this);
	m_yaw_sub   = n.subscribe(yaw_topic   + "_cmd", 1000, &GazeboHardwareNode::yawCallback, this);
	m_depth_sub = n.subscribe(depth_topic + "_cmd", 1000, &GazeboHardwareNode::depthCallback, this);
	m_surge_sub = n.subscribe(surge_topic + "_cmd", 1000, &GazeboHardwareNode::surgeCallback, this);
	m_sway_sub  = n.subscribe(sway_topic  + "_cmd", 1000, &GazeboHardwareNode::swayCallback, this);
	
	m_roll_pub = n.advertise<std_msgs::Float64>(roll_topic, 1000);
	m_pitch_pub = n.advertise<std_msgs::Float64>(pitch_topic, 1000);
	m_yaw_pub = n.advertise<std_msgs::Float64>(yaw_topic, 1000);
	m_depth_pub = n.advertise<std_msgs::Float64>(depth_topic, 1000);

		
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
		
	//add yaw,pitch,roll commands to our thrusters
	m_thruster_commands[0].data -= m_yaw_command;
	m_thruster_commands[1].data += m_yaw_command;
	m_thruster_commands[2].data += m_yaw_command;
	m_thruster_commands[3].data -= m_yaw_command;
	
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

	//surge commands
	m_thruster_commands[0].data += m_surge_command;
	m_thruster_commands[1].data += m_surge_command;
	m_thruster_commands[2].data += m_surge_command;
	m_thruster_commands[3].data += m_surge_command;
	
	//sway commands
	m_thruster_commands[0].data -= m_sway_command;
	m_thruster_commands[1].data += m_sway_command;
	m_thruster_commands[2].data -= m_sway_command;
	m_thruster_commands[3].data += m_sway_command;
	
	
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

void GazeboHardwareNode::surgeCallback(const std_msgs::Float64::ConstPtr& msg){
	m_surge_command = msg->data; 
}

void GazeboHardwareNode::swayCallback(const std_msgs::Float64::ConstPtr& msg){
	m_sway_command = msg->data; 
}



//--------------------------------------------------------------------------
// Data callbacks, these take info from the simulator and pass it on in a way
// that should appear identical to how the real qubo will present it's data

void GazeboHardwareNode::orientCallback(const nav_msgs::Odometry::ConstPtr &msg){
	
    
    //this is a little clunky, but it's the best way I could find to convert from a quaternion to Euler Angles
	tf::Quaternion q(msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, msg->pose.pose.orientation.z, msg->pose.pose.orientation.w);
	tf::Matrix3x3 m(q);
	std_msgs::Float64 roll, pitch, yaw;
	m.getRPY(roll.data, pitch.data, yaw.data); //roll pitch and yaw are populated
	
	
	//m_orient_pub.publish(*msg);
	m_roll_pub.publish(roll);
	m_pitch_pub.publish(pitch);
	m_yaw_pub.publish(yaw);

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
