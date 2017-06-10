#include "kalman_filter.h"

namespace Controls {

KalmanFilter::KalmanFilter() {
    dvl_sub = nh->subscribe("/qubo/dvl", 1, &KalmanFilter::dvl_callback, this);
    ahrs_sub = nh->subscribe("/qubo/ahrs", 1, &KalmanFilter::ahrs_callback, this);
    imu_sub = nh->subscribe("/qubo/imu", 1, &KalmanFilter::imu_callback, this);
}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::run() {
    while (ros::ok()) {
        ros::spinOnce();
    }
}

void KalmanFilter::dvl_callback(const geometry_msgs::TwistWithCovarianceStamped& dvl_data) {
    
}

void KalmanFilter::ahrs_callback(const sensor_msgs::Imu& ahrs_data) {

}

void KalmanFilter::imu_callback(const sensor_msgs::Imu& imu_data) {
    
}

}
