#include "depth_controller.h"

using namespace std;
using namespace ros; 

DepthController::DepthController(NodeHandle n) {

	//TODO how accurate is ros::Time going to be for control purposes?
	m_prev_time = ros::Time::now();
	
    // Set up publishers and subscribers
	string qubo_namespace = "/qubo/";
	
	string sensor_topic = qubo_namespace + "depth";
	m_sensor_sub = n.subscribe(sensor_topic, 1000, &DepthController::sensorCallback, this);
	
	string command_topic = qubo_namespace + "depth_command";
	m_command_pub = n.advertise<std_msgs::Float64>(command_topic, 1000);

	m_depth_command.data = 5;
    
}

DepthController::~DepthController() {}

void DepthController::update() {
	//update our commanded and measured depth.
	ros::spinOnce();
	
	// Calculate time passed since previous loop
	ros::Duration dt = ros::Time::now() - m_prev_time;
	m_prev_time = ros::Time::now();
    
	//calculate error, update integrals and derivatives of the error
	m_error            = m_desired_depth - m_current_depth; //proportional term
	m_error_integral  += m_error * dt.toSec(); //integral term
	m_error_derivative = (m_error -m_prev_error)/dt.toSec();

	//store the previous error
	m_prev_error = m_error;

	//sum everything weighted by the given gains. 
	m_depth_command.data = (m_kp*m_error) + (m_ki*m_error_integral) + (m_kd*m_error_derivative); 
	m_command_pub.publish(m_depth_command);
	
}

void DepthController::sensorCallback(const std_msgs::Float64::ConstPtr& msg) {
	m_current_depth = msg->data;
	
}



