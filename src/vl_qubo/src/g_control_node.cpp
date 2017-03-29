#include "g_control_node.h"

using namespace std;
using namespace ros;

GControlNode::GControlNode(ros::NodeHandle n, string node_name, string fused_pose_topic)
    :_node_name(node_name){

    string input_pose = "/basic_qubo/pose_gt";
    
    _orient_sub = n.subscribe(input_pose, 1000, &GControlNode::orientCallback, this);
    _orient_pub = n.advertise<sensor_msgs::Imu>(fused_pose_topic.c_str(),1000);

}

GControlNode::~GControlNode(){}

void GControlNode::update(){
    spinOnce(); //get all the callbacks 
    
    //!!!! sum roll/pitch/yaw into thruster commands here.
    
    
}

void GControlNode::orientCallback(const nav_msgs::Odometry::ConstPtr &msg){

    //may have to convert to quaternions here..
    // ROS_INFO("Seq: [%d]", msg->header.seq);
    // ROS_INFO("Position-> x: [%f], y: [%f], z: [%f]", msg->pose.pose.position.x,msg->pose.pose.position.y, msg->pose.pose.position.z);
    //ROS_INFO("Orientation-> x: [%f], y: [%f], z: [%f], w: [%f]", msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, msg->pose.pose.orientation.z, msg->pose.pose.orientation.w);
    //ROS_INFO("Vel-> Linear: [%f], Angular: [%f]", msg->twist.twist.linear.x,msg->twist.twist.angular.z);


    //add this to the header? idk
    //tf::Quaternion ahrsData(msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, msg->pose.pose.orientation.z, msg->pose.pose.orientation.w);

    
    _orient_pub.publish(*msg);
    
}

void GControlNode::pitchCallback(const std_msgs::Float64::ConstPtr& msg){
    _pitch_command = msg->data;
}


void GControlNode::yawCallback(const std_msgs::Float64::ConstPtr& msg){
    _yaw_command = msg->data;
}


void GControlNode::rollCallback(const std_msgs::Float64::ConstPtr& msg){
    _roll_command = msg->data;
}

