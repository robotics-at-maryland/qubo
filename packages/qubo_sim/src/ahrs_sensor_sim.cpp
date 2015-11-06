#include "ros/ros.h"
#include "std_msgs/String.h"
#include "ram_msgs/Sim_AHRS.h"
#include "sensor_msgs/Imu.h"

#include <sstream>

void data_callback(const sensor_msgs::Imu::ConstPtr &msg) {
    ROS_INFO("I heard: [%f]", msg->orientation.x);
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "simulated_ahrs_sensor");
    ros::NodeHandle n;

    ros::Subscriber ahrs_sub = n.subscribe("/g500/imu", 100, data_callback);
    
    ros::spin();

    return 0;
}
