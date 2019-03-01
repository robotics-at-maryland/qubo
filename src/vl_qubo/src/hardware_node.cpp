#include "hardware_node.h"

using namespace std;
using namespace ros;

HardwareNode::HardwareNode(ros::NodeHandle n, string node_name)
  :m_node_name(node_name), m_thruster_commands(NUM_THRUSTERS), m_thruster_pubs(NUM_THRUSTERS) {

  //might need service for toggling position and velocity here at some point

  string hn_namespace = "/hardware_node/";
  string cont_namespace = "/qubo/";

  string roll_topic = cont_namespace + "roll";
  string pitch_topic = cont_namespace + "pitch";
  string yaw_topic = cont_namespace + "yaw";
  string depth_topic = cont_namespace + "depth";
  string surge_topic = cont_namespace + "surge"; //forward
  string sway_topic = cont_namespace + "sway"; //sideways

  m_roll_sub  = n.subscribe(roll_topic  + "_cmd", 1000, &HardwareNode::rollCallback, this);
  m_pitch_sub = n.subscribe(pitch_topic + "_cmd", 1000, &HardwareNode::pitchCallback, this);
  m_yaw_sub   = n.subscribe(yaw_topic   + "_cmd", 1000, &HardwareNode::yawCallback, this);
  m_depth_sub = n.subscribe(depth_topic + "_cmd", 1000, &HardwareNode::depthCallback, this);
  m_surge_sub = n.subscribe(surge_topic + "_cmd", 1000, &HardwareNode::surgeCallback, this);
  m_sway_sub  = n.subscribe(sway_topic  + "_cmd", 1000, &HardwareNode::swayCallback, this);

  //register the thruster topics, we have 8
  string t_topic =  hn_namespace + "thrusters";
            
  for(int i = 0; i < NUM_THRUSTERS; i++){
        m_thruster_commands[i].data = 0;

        t_topic = hn_namespace + "thrusters/";
        t_topic += to_string(i);
		t_topic += "/input";
		
        m_thruster_pubs[i] = n.advertise<uuv_gazebo_ros_plugins_msgs::FloatStamped>(t_topic, 1000);
        cout << t_topic << endl;

    }
}

HardwareNode::~HardwareNode(){}

void HardwareNode::update(){
  spinOnce();

  for(int i = 0; i < NUM_THRUSTERS; i++){
    m_thruster_commands[i].data = 0;		
  }
	
  //surge, sway, yaw thrusters
  m_thruster_commands[0].data += (m_surge_command - m_yaw_command - m_sway_command);
  m_thruster_commands[1].data += (m_surge_command + m_yaw_command + m_sway_command);
  m_thruster_commands[2].data += (m_surge_command + m_yaw_command - m_sway_command);
  m_thruster_commands[3].data += (m_surge_command - m_yaw_command + m_sway_command);
	
  //depth, pitch, roll thrusters
  m_thruster_commands[4].data += (m_depth_command + m_pitch_command + m_roll_command);
  m_thruster_commands[5].data += (m_depth_command + m_pitch_command - m_roll_command);
  m_thruster_commands[6].data += (m_depth_command - m_pitch_command - m_roll_command);
  m_thruster_commands[7].data += (m_depth_command - m_pitch_command + m_roll_command);
	
  for(int i = 0; i < NUM_THRUSTERS; i++){
    m_thruster_pubs[i].publish(m_thruster_commands[i]);
  }
	
}

void HardwareNode::yawCallback(const std_msgs::Float64::ConstPtr& msg) {
  m_yaw_command = msg->data;
}

void HardwareNode::pitchCallback(const std_msgs::Float64::ConstPtr& msg) {
  m_pitch_command = msg->data;
}

void HardwareNode::rollCallback(const std_msgs::Float64::ConstPtr& msg) {
  m_roll_command = msg->data;
}

void HardwareNode::depthCallback(const std_msgs::Float64::ConstPtr& msg) {
  m_depth_command = msg->data;
}

void HardwareNode::surgeCallback(const std_msgs::Float64::ConstPtr& msg) {
  m_surge_command = msg->data;
}

void HardwareNode::swayCallback(const std_msgs::Float64::ConstPtr& msg) {
  m_sway_command = msg->data;
}


  
