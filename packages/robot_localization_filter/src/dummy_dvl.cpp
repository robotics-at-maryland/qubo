/**
 *  This is a simple publisher that publishes "dummy" DVL data to the 
 *  robot_localization package.
 */
#include <ros/ros.h>
#include <geometry_msgs/TwistWithCovarianceStamped.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    // Initialize the "dummy_dvl" node
    ros::init(argc, argv, "dvl");
    
    // Create a NodeHandle object called dvl
    ros::NodeHandle dvl;
    
    // Tell ROS master this node is publishing Twist messages to topic
    // qubo_localization/twist
    ros::Publisher dvl_data = dvl.advertise<geometry_msgs::TwistWithCovarianceStamped>("dvl/twist", 1000);
    
    // Specify the rate at which messages will be published                
    ros::Rate rate(30);
    
    while(ros::ok()) {
        // Create a TwistWithCovarianceStamped object
        geometry_msgs::TwistWithCovarianceStamped msg;
        
        // Set the data included in the twist message
        msg.header.frame_id = "base_link";
        msg.header.stamp = ros::Time::now();
        
        msg.twist.twist.linear.x = 1;
        msg.twist.twist.linear.y = 0;
        msg.twist.twist.linear.z = 0;
        
        msg.twist.covariance = {1e-9, 0, 0, 0, 0, 0,
                                0, 1e-9, 0, 0, 0, 0,
                                0, 0, 1e-9, 0, 0, 0,
                                0, 0, 0, 1e-9, 0, 0,
                                0, 0, 0, 0, 1e-9, 0,
                                0, 0, 0, 0, 0, 1e-9};
        
        // Send the message
        dvl_data.publish(msg);
        
        rate.sleep();
    }
    
    return 0;
}
