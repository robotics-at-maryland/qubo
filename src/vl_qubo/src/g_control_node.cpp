#include "g_control_node.h"

using namespace std;
using namespace ros;

GControlNode::GControlNode(ros::NodeHandle n, string node_name, string fused_pose_topic)
    :_node_name(node_name){

    string input_pose = "/basic_qubo/pose_gt";
    
    _orientSub = n.subscribe(input_pose, 1000, &GControlNode::orientCallback, this);
    _orientPub = n.advertise<sensor_msgs::Imu>(fused_pose_topic.c_str(),1000);

}

GControlNode::~GControlNode(){}

void GControlNode::update(){
    spinOnce();
    _orientPub.publish(_fusedPose);

    

}

void GControlNode::orientCallback(const nav_msgs::Odometry::ConstPtr msg){
    //may have to convert to quaternions here..
    ROS_INFO("Seq: [%d]", msg->header.seq);
    ROS_INFO("Position-> x: [%f], y: [%f], z: [%f]", msg->pose.pose.position.x,msg->pose.pose.position.y, msg->pose.pose.position.z);
    ROS_INFO("Orientation-> x: [%f], y: [%f], z: [%f], w: [%f]", msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, msg->pose.pose.orientation.z, msg->pose.pose.orientation.w);
    ROS_INFO("Vel-> Linear: [%f], Angular: [%f]", msg->twist.twist.linear.x,msg->twist.twist.angular.z);
    
    // _fusedPose.header.stamp = ros::Time::now();
    // _fusedPose.header.seq = ++id;
	// _fusedPose.header.frame_id = "orientation";

	// _fusedPose.orientation.x = _ahrs_data.quaternion[0];
	// _fusedPose.orientation.y = _ahrs_data.quaternion[1];
	// _fusedPose.orientation.z = _ahrs_data.quaternion[2];
	// _fusedPose.orientation.w = _ahrs_data.quaternion[3];
	// // the -1's imply we don't know the covariance
	// _fusedPose.orientation_covariance[0] = -1;

	// _fusedPose.angular_velocity.x = _ahrs_data.gyroX;
	// _fusedPose.angular_velocity.y = _ahrs_data.gyroY;
	// _fusedPose.angular_velocity.z = _ahrs_data.gyroZ;
	// _fusedPose.angular_velocity_covariance[0] = -1;

	// _fusedPose.linear_acceleration.x = _ahrs_data.accelX;
	// _fusedPose.linear_acceleration.y = _ahrs_data.accelY;
	// _fusedPose.linear_acceleration.z = _ahrs_data.accelZ;
	// _fusedPosey.linear_acceleration_covariance[0] = -1;
    

}
