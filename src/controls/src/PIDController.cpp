#include "PIDController.h"

PIDController::PIDController() {}
PIDController::~PIDController() {}

void PIDController::robot_state_callback(const nav_msgs::OdometryConstPtr& current_state) {
    // Calculate time passed since previous loop
    ros::Duration dt;
    if (!prev_time.isZero()) {
        dt = ros::Time::now() - prev_time;
        prev_time = ros::Time::now();
    } else {
        prev_time = ros::Time::now();
        return;
    }
    
    // Save previous errors
    prev_error_x = error_x;
    prev_error_y = error_y;
    prev_error_z = error_z;
    
    // Calculate proportional error
    float error_x = desired_x - current_state->pose.pose.position.x;
    float error_y = desired_y - current_state->pose.pose.position.y;
    float error_z = desired_z - current_state->pose.pose.position.z;
    
    // Calculate integral error
    integral_error_x += error_x * dt.toSec();
    integral_error_y += error_y * dt.toSec();
    integral_error_z += error_z * dt.toSec();
    
    // Calculate derivative error
    float derivative_error_x = (error_x - prev_error_x) / dt.toSec();
    float derivative_error_y = (error_y - prev_error_y) / dt.toSec();
    float derivative_error_z = (error_z - prev_error_z) / dt.toSec();
    
    // Combine errors to get total control effort
    float control_effort_x = Kp_x * error_x + Ki_x * integral_error_x + Kd_x * derivative_error_x;
    float control_effort_y = Kp_y * error_y + Ki_y * integral_error_y + Kd_y * derivative_error_y;
    float control_effort_z = Kp_z * error_z + Ki_z * integral_error_z + Kd_z * derivative_error_z;
    
    // Apply upper limit to control efforts
    if (control_effort_x > upper_limit) {
        control_effort_x = upper_limit;
    } else if (control_effort_x < lower_limit) {
        control_effort_x = lower_limit;
    }
    
    if (control_effort_y > upper_limit) {
        control_effort_y = upper_limit;
    } else if (control_effort_y < lower_limit) {
        control_effort_y = lower_limit;
    }
    
    if (control_effort_z > upper_limit) {
        control_effort_z = upper_limit;
    } else if (control_effort_z < lower_limit) {
        control_effort_z = lower_limit;
    }
    
    /* Publish control efforts to thrusters */
}

void PIDController::setpoint_callback(const nav_msgs::OdometryConstPtr& desired_state) {
    desired_x = desired_state->pose.pose.position.x;
    desired_y = desired_state->pose.pose.position.y;
    desired_z = desired_state->pose.pose.position.z;
}

int main(int argc, char** argv) {
}
