/**
 *  This is a simple publisher that publishes "dummy" IMU data to the 
 *  robot_localization package.
 */
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    // Initialize the "dummy_imu" node
    ros::init(argc, argv, "imu");
    
    // Create a NodeHandle object called imu
    ros::NodeHandle imu;
    
    // Tell ROS master this node is publishing Imu messages to topic
    // qubo_localization/imu/data
    ros::Publisher imu_data = imu.advertise<sensor_msgs::Imu>("imu/imu/data", 1000);
    
    // Specify the rate at which messages will be published                
    ros::Rate rate(30);
    
    while(ros::ok()) {
        // Create a TwistWithCovarianceStamped object
        sensor_msgs::Imu msg;
        
        // Set the data included in the Imu message  
        msg.header.stamp = ros::Time::now();
        msg.header.frame_id = "odom";
        
        msg.linear_acceleration.x = 0;
        msg.linear_acceleration.y = 0;
        msg.linear_acceleration.z = 0;
        
        // Send the message
        imu_data.publish(msg);
        
        rate.sleep();
    }
    
    return 0;
}
