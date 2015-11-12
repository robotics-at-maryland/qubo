#include "ros/ros.h"
#include "std_msgs/String.h"
#include "sensor_msgs/Imu.h"

#include <sstream>

sensor_msgs::Imu ahrs_msg;
ros::Publisher ahrs_pub;

void data_callback(const sensor_msgs::Imu::ConstPtr &imu_msg) {
    ahrs_msg.orientation.x = imu_msg->orientation.x;
    ahrs_msg.orientation.y = imu_msg->orientation.y;
    ahrs_msg.orientation.z = imu_msg->orientation.z;

    ahrs_pub.publish(ahrs_msg);
    ros::spin();
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "simulated_ahrs_sensor");
    ros::NodeHandle n;

    ahrs_pub = n.advertise<sensor_msgs::Imu>("/qubo/ahrs", 100);
    ros::Subscriber ahrs_sub = n.subscribe("/g500/imu", 100, data_callback);
    
    ros::spin();
    return 0;
}
