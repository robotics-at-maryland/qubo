/**
 *  This is a simple publisher that publishes "dummy" AHRS data to the 
 *  robot_localization package.
 */
#include <ros/ros.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    // Initialize the "dummy_ahrs" node
    ros::init(argc, argv, "ahrs");
    
    // Create a NodeHandle object called ahrs
    ros::NodeHandle ahrs;
    
    // Tell ROS master this node is publishing Pose messages to topic qubo_localization/pose
    ros::Publisher ahrs_data = ahrs.advertise<geometry_msgs::PoseWithCovarianceStamped>("ahrs/pose", 1000);
    
    // Specify the rate at which messages will be published                
    ros::Rate rate(30);
    
    while(ros::ok()) {
        // Create a PoseWithCovarianceStamped object
        geometry_msgs::PoseWithCovarianceStamped msg;
        
        // Set the data included in the twist message
        msg.header.frame_id = "odom";
        msg.header.stamp = ros::Time::now();
        
        msg.pose.pose.orientation.x = 0;
        msg.pose.pose.orientation.y = 0;
        msg.pose.pose.orientation.z = 0;
        msg.pose.pose.orientation.w = 1.0;
        
        msg.pose.covariance = {1e-9, 0, 0, 0, 0, 0,
                               0, 1e-9, 0, 0, 0, 0,
                               0, 0, 1e-9, 0, 0, 0,
                               0, 0, 0, 1e-9, 0, 0,
                               0, 0, 0, 0, 1e-9, 0,
                               0, 0, 0, 0, 0, 1e-9};
        
        // Send the message
        ahrs_data.publish(msg);
        
        rate.sleep();
    }
    
    return 0;
}
