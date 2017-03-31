#include "g_control_node.h"

using namespace std;
using namespace ros;

GControlNode::GControlNode(ros::NodeHandle n, string node_name, string fused_pose_topic)
    :_node_name(node_name), _thruster_values(NUM_THRUSTERS), _thruster_pubs(NUM_THRUSTERS) {
    
    qubo_name = "/basic_qubo/";
    
    string input_pose = qubo_name + "pose_gt";
    string yaw_topic = qubo_name + "controller/yaw";
    string pitch_topic = qubo_name + "controller/pitch";
    string roll_topic = qubo_name + "controller/roll";
    
    _orient_sub = n.subscribe(input_pose, 1000, &GControlNode::orientCallback, this);
    _orient_pub = n.advertise<sensor_msgs::Imu>(fused_pose_topic.c_str(),1000);

    _yaw_sub = n.subscribe(yaw_topic, 1000, &GControlNode::yawCallback, this);
    _pitch_sub = n.subscribe(pitch_topic, 1000, &GControlNode::pitchCallback, this);
    _roll_sub = n.subscribe(roll_topic, 1000, &GControlNode::rollCallback, this);


    string t_topic = "/basic_qubo/thruster";
            
        
    for(int i = 0; i < NUM_THRUSTERS; i++){
        _thruster_values[i] = 0;

        t_topic = "/basic_qubo/thruster";
        t_topic += to_string(i);
        
        _thruster_pubs[i] = n.advertise<std_msgs::Float64>(t_topic, 1000);
        cout << t_topic << endl;

    }

}

GControlNode::~GControlNode(){}

void GControlNode::update(){
    spinOnce(); //get all the callbacks

    ROS_ERROR("yaw pitch roll = %d %d %d"); 
    
    //!!!! sum roll/pitch/yaw into thruster commands here.
    
    
}

void GControlNode::orientCallback(const nav_msgs::Odometry::ConstPtr &msg){

    //may have to convert to quaternions here..
    // ROS_INFO("Seq: [%d]", msg->header.seq);
    // ROS_INFO("Position-> x: [%f], y: [%f], z: [%f]", msg->pose.pose.position.x,msg->pose.pose.position.y, msg->pose.pose.position.z);
    //ROS_INFO("Orientation-> x: [%f], y: [%f], z: [%f], w: [%f]", msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, msg->pose.pose.orientation.z, msg->pose.pose.orientation.w);
    //ROS_INFO("Vel-> Linear: [%f], Angular: [%f]", msg->twist.twist.linear.x,msg->twist.twist.angular.z);


    //add this to the header?

    //tf::Quaternion ahrsData(msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, msg->pose.pose.orientation.z, msg->pose.pose.orientation.w);

    
    _orient_pub.publish(*msg);

}


void GControlNode::yawCallback(const std_msgs::Float64::ConstPtr& msg){
    _yaw_command = msg->data;
}

void GControlNode::pitchCallback(const std_msgs::Float64::ConstPtr& msg){
    _pitch_command = msg->data;
}

void GControlNode::rollCallback(const std_msgs::Float64::ConstPtr& msg){
    _roll_command = msg->data;
}

