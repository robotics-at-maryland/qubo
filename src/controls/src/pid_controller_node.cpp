#include "pid_controller.h"

#include <ros/ros.h>

int main(int argc, char **argv) {
    ros::init(argc, argv, "pid_controller_node");
    ros::NodeHandle nh;

    Controls::PIDController pid(nh);
    pid.run();

    return 0;
}
