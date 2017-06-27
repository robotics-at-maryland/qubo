#include "qscu_control_node.h"

using namespace std;

QSCUControlNode::QSCUControlNode(ros::NodeHandle n, string node_name, string device_file)
	:m_node_name(node_name), qscu(device_file, B115200) {

	string qubo_namespace = "/qubo/";

	string embedded_status_topic = qubo_namespace + "embedded_status";
	m_status_pub = n.advertise<ram_msgs::Status>(embedded_status_topic, 1000);

	// Using `new` means we have to manually free this thing
	qubobus_loop = n.createTimer(ros::Duration(0.05), &QSCUControlNode::QubobusCallback, this);
	qubobus_loop.start();
}

QSCUControlNode::~QSCUControlNode(){
	qubobus_loop.stop();
}

void QSCUControlNode::update(){
	ros::spin();
}

void QSCUControlNode::QubobusCallback(const ros::TimerEvent& event){ 

	if ( !qscu.isOpen() ) {
		try {
			qscu.openDevice();
		} catch ( const QSCUException ex ) {
			ROS_ERROR("Unable to connect to the embedded system at the specified location");
			ROS_ERROR("=> %s", ex.what() );
			return;
		}
	}
	Transaction t_e = tEmbeddedStatus;
	struct Embedded_Status e_s;

	try {
		qscu.sendMessage(&t_e, NULL, &e_s);
		m_status_msg.uptime = e_s.uptime;
		m_status_msg.memory_capacity = e_s.mem_capacity;
		m_status_pub.publish(m_status_msg);
	} catch ( const QSCUException ex ) {
		ROS_ERROR("Error reading the embedded system status");
		ROS_ERROR("=> %s", ex.what() );
		return;
	}
}
