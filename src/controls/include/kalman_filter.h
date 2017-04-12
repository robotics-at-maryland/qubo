#ifndef KALMAN_FILTER_H
#define KALMAN_FILTER_H

#include <ros/ros.h>
#include <geometry_msgs::TwistWithCovarianceStamped>
#include <sensor_msgs::Imu>
#include <Eigen/Dense>

namespace Controls {

class KalmanFilter {
    public:
        KalmanFilter(ros::NodeHandle *nh);
        ~KalmanFilter();

        void run();

        void dvl_callback(const geometry_msgs::TwistWithCovarianceStamped& dvl_data);
        void ahrs_callback(const sensor_msgs::Imu& ahrs_data);
        void imu_callback(const sensor_msgs::Imu& ahrs_data);

    private:
        ros::Subscriber dvl_sub;
        ros::Subscriber ahrs_sub;
        ros::Subscriber imu_sub;

        Eigen::VectorXd
};

} // namespace Controls

#endif // KALMAN_FILTER_H
