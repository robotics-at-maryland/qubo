#include "kalman_filter.h"

#include <ros/ros.h>

int main(int argc, char **argv) {
    ros::init(argc, argv, "kalman_filter_node");
    ros::NodeHandle nh;

    Controls::KalmanFilter ekf(&nh);
    ekf.run();

    return 0;
}

