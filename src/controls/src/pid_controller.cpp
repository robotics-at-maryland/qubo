#include "pid_controller.h"

using namespace std;
using namespace ros;


PIDController::PIDController(NodeHandle n, string control_topic) {

	//TODO how accurate is ros::Time going to be for control purposes?
	m_prev_time = ros::Time::now();
	
    // Set up publishers and subscribers
	string qubo_namespace = "/qubo/";
	
	string sensor_topic = qubo_namespace + control_topic;
	m_sensor_sub = n.subscribe(sensor_topic, 1000, &PIDController::sensorCallback, this);
	
	string command_topic = qubo_namespace + control_topic + "_cmd";
	m_command_pub = n.advertise<std_msgs::Float64>(command_topic, 1000);

	m_command_msg.data = 5;

	
	f = boost::bind(&PIDController::configCallback, this, _1, _2);
	server.setCallback(f);
	
    
}

PIDController::~PIDController() {}

void PIDController::update() {
	//update our commanded and measured depth.
	ros::spinOnce();
	
	// Calculate time passed since previous loop
	ros::Duration dt = ros::Time::now() - m_prev_time;
	m_prev_time = ros::Time::now();
    
	//calculate error, update integrals and derivatives of the error
	m_error            = m_desired - m_current; //proportional term
	m_error_integral  += m_error * dt.toSec(); //integral term
	m_error_derivative = (m_error -m_prev_error)/dt.toSec();

	//store the previous error
	m_prev_error = m_error;

	//sum everything weighted by the given gains. 
	m_command_msg.data = (m_kp*m_error) + (m_ki*m_error_integral) + (m_kd*m_error_derivative); 
	m_command_pub.publish(m_command_msg);
	
}

void PIDController::sensorCallback(const std_msgs::Float64::ConstPtr& msg) {
	m_current = msg->data;
	
}


void PIDController::configCallback(controls::TestConfig &config, uint32_t level) {
	ROS_INFO("Reconfigure Request: %f %f %f %f", config.kp, config.ki, config.kd, config.target);
	m_kp = config.kp;
	m_ki = config.ki;
	m_kd = config.kd;
	m_desired = config.target;
	
}



