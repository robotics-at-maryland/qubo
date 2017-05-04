#ifndef DEPTH_CONTROLLER_H
#define DEPTH_CONTROLLER_H

#include <ros/ros.h>
#include "std_msgs/Float64.h"


#include <dynamic_reconfigure/server.h>
#include <controls/TestConfig.h>


class PIDController {
    public:
    PIDController(ros::NodeHandle n, std::string topic_name);
    ~PIDController();

	void update();
	
    protected:
    
    ros::Time m_prev_time; 

    ros::Subscriber m_sensor_sub;
    void sensorCallback(const std_msgs::Float64::ConstPtr& msg);
    double m_current;

	ros::Publisher m_command_pub;
    //will want this eventually
    //void commandCallback(const std_msgs::Float64::ConstPtr& msg);
    std_msgs::Float64  m_command_msg;

	double m_desired = 5;

    std::string m_control_topic;
    
    //P,I, and D terms, as it where. 
    double m_error;
    double m_error_integral;
    double m_error_derivative; 

	double m_prev_error;

    //gains
    double m_kp = 1;
    double m_ki = 0;
    double m_kd = 0;

	dynamic_reconfigure::Server<controls::TestConfig> server;
	dynamic_reconfigure::Server<controls::TestConfig>::CallbackType f;

    void configCallback(controls::TestConfig &config, uint32_t level);

	
};


#endif //PID_CONTROLLER_H


	
