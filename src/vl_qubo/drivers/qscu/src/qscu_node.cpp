#include "qscu_node.h"

using namespace std;

const bool QSCUNode::QMsg::operator<(const QMsg& obj) const {
	return false;
}

// bool QSCUNode::QMsgCompare(std::pair<int, QMsg> p1, std::pair<int, QMsg> p2) {
//	return p1.first < p2.first;
// }

QSCUNode::QSCUNode(ros::NodeHandle n, string node_name, string device_file)
	:m_node_name(node_name), qscu(device_file, B115200), KEEPALIVE_SEND_PERIOD(1000) {

	string qubo_namespace = "/qubo/";

	string embedded_status_topic = qubo_namespace + "embedded_status";
	m_status_pub = n.advertise<ram_msgs::Status>(embedded_status_topic, 1000);


	/**
	 * This stuff is almost exactly the same as it is in the Gazebo node
	 */
	string roll_topic	= qubo_namespace + "roll";
	string pitch_topic	= qubo_namespace + "pitch";
	string yaw_topic	= qubo_namespace + "yaw";
	string depth_topic	= qubo_namespace + "depth"; //sometimes called heave
	string surge_topic	= qubo_namespace + "surge"; //"forward" translational motion
	string sway_topic	= qubo_namespace + "sway";   //"sideways" translational motion

	m_roll_sub  = n.subscribe(roll_topic  + "_cmd", 1000, &QSCUNode::rollCallback, this);
	m_pitch_sub = n.subscribe(pitch_topic + "_cmd", 1000, &QSCUNode::pitchCallback, this);
	m_yaw_sub   = n.subscribe(yaw_topic   + "_cmd", 1000, &QSCUNode::yawCallback, this);
	m_depth_sub = n.subscribe(depth_topic + "_cmd", 1000, &QSCUNode::depthCallback, this);
	m_surge_sub = n.subscribe(surge_topic + "_cmd", 1000, &QSCUNode::surgeCallback, this);
	m_sway_sub  = n.subscribe(sway_topic  + "_cmd", 1000, &QSCUNode::swayCallback, this);

	m_thruster_speeds.resize(8);

	/**
	 * Creates a Timer object, which will trigger every `Duration` amount of time
	 * to allow us to have a bit more accuracy in the time between updates on Qubobus
	 */
	qubobus_loop		   = n.createTimer(ros::Duration(0.35), &QSCUNode::QubobusCallback, this);
	qubobus_incoming_loop  = n.createTimer(ros::Duration(0.1), &QSCUNode::QubobusIncomingCallback, this);
	// qubobus_status_loop = n.createTimer(ros::Duration(5), &QSCUNode::QubobusStatusCallback, this);
	qubobus_thruster_loop  = n.createTimer(ros::Duration(0.3), &QSCUNode::QubobusThrusterCallback, this);
	// qubobus_depth_loop  = n.createTimer(ros::Duration(4), &QSCUNode::QubobusDepthCallback, this);

	qubobus_loop.start();
	qubobus_incoming_loop.start();
	// qubobus_depth_loop.start();
	// qubobus_status_loop.start();
}

QSCUNode::~QSCUNode(){
	qubobus_loop.stop();
}

void QSCUNode::update(){
	ros::spin();
}

void QSCUNode::QubobusThrusterCallback(const ros::TimerEvent& event){

	if (!thruster_update || thruster_update) {
		// update the actual commands from the buffer, so these don't get changed in the middle of running
		m_yaw_command	= m_yaw_command_buffer;
		m_pitch_command = m_pitch_command_buffer;
		m_roll_command	= m_roll_command_buffer;
		m_depth_command = m_depth_command_buffer;
		m_surge_command = m_surge_command_buffer;
		m_sway_command	= m_sway_command_buffer;

		// Calculate the values for the thrusters we need to change
		m_thruster_speeds[0] = -m_yaw_command + m_surge_command - m_sway_command;
		m_thruster_speeds[1] = +m_yaw_command + m_surge_command + m_sway_command;
		m_thruster_speeds[2] = +m_yaw_command + m_surge_command - m_sway_command;
		m_thruster_speeds[3] = -m_yaw_command + m_surge_command + m_sway_command;

		m_thruster_speeds[4] = ( m_pitch_command + m_roll_command) + m_depth_command;
		m_thruster_speeds[5] = ( m_pitch_command - m_roll_command) + m_depth_command;
		m_thruster_speeds[6] = (-m_pitch_command - m_roll_command) + m_depth_command;
		m_thruster_speeds[7] = (-m_pitch_command + m_roll_command) + m_depth_command;

		// Create the message and add it to the queue
		QMsg q_msg;
		q_msg.type = tThrusterSet;
		auto tmp = std::make_shared<struct Thruster_Set>();
		for (uint8_t i = 0; i < 8; i++) {
			tmp->throttle[i] = m_thruster_speeds[i];
		}
		q_msg.payload = tmp;
		q_msg.reply = nullptr;
		m_outgoing.push(make_pair(THRUSTER_PRIORITY, q_msg));
		//	QMsg q_msg;
		//	q_msg.type = tThrusterSet;
		//	q_msg.payload = std::make_shared<struct Thruster_Set>( (struct Thruster_Set) {
		//			.throttle = m_thruster_speeds[i],
		//				.thruster_id = i,
		//				});
		// }

		thruster_update = false;
		ROS_ERROR("Sending message");
	}

}

void QSCUNode::QubobusIncomingCallback(const ros::TimerEvent& event){
	while (!m_incoming.empty()) {
		QMsg msg = m_incoming.front();
		if (msg.type.id == tEmbeddedStatus.id){
			std::shared_ptr<struct Embedded_Status> e_s =
				std::static_pointer_cast<struct Embedded_Status>(msg.reply);
			ROS_ERROR("Uptime: %i, Mem: %f", e_s->uptime, e_s->mem_capacity);
		} else if(msg.type.id == tDepthStatus.id) {
			std::shared_ptr<struct Depth_Status> d_s =
				std::static_pointer_cast<struct Depth_Status>(msg.reply);
			ROS_ERROR("Depth Reading: %f", d_s->depth_m);
		}

		m_incoming.pop();
	}
}

void QSCUNode::QubobusStatusCallback(const ros::TimerEvent& event){
	QMsg q_msg;
	q_msg.type = tEmbeddedStatus;
	q_msg.payload = nullptr;
	q_msg.reply = std::make_shared<struct Embedded_Status>();
	m_outgoing.push(make_pair(STATUS_PRIORITY, q_msg));
}

void QSCUNode::QubobusDepthCallback(const ros::TimerEvent& event){
	QMsg q_msg;
	q_msg.type = tDepthStatus;
	q_msg.payload = nullptr;
	q_msg.reply = std::make_shared<struct Depth_Status>();
	m_outgoing.push(make_pair(THRUSTER_PRIORITY, q_msg));
}


void QSCUNode::QubobusCallback(const ros::TimerEvent& event){

	if ( !qscu.isOpen() ) {
		try {
			qscu.openDevice();
		} catch ( const QSCUException& ex ) {
			ROS_ERROR("Unable to connect to the embedded system at the specified location");
			ROS_ERROR("=> %s", ex.what() );
			return;
		}
	}

	try {
		if (m_outgoing.empty()) {
			// We only want to send a keepalive every so often
			if ((m_last_keepalive_sent_time + KEEPALIVE_SEND_PERIOD) < std::chrono::steady_clock::now()) {
				qscu.keepAlive();
				m_last_keepalive_sent_time = std::chrono::steady_clock::now();
			}
		} else {
			// Try to clear the buffer of messages
			// Don't know what to do when there are too many messages
			while (!m_outgoing.empty()){
				QMsg msg = m_outgoing.top().second;
				// struct Thruster_Set* test = (struct Thruster_Set*) msg.payload.get();
				// ROS_ERROR("out: %f", test->throttle);
				qscu.sendMessage(&msg.type, msg.payload.get(), msg.reply.get());
				m_incoming.push(msg);
				m_outgoing.pop();
			}
		}
	} catch ( const QSCUException& ex ) {
		ROS_ERROR("Error reading the embedded system status");
		ROS_ERROR("=> %s", ex.what() );
		try {
			qscu.connect();
		} catch ( const QSCUException& ex ) {
			ROS_ERROR("Unable to connect to the Tiva");
		}
		return;
	}
}

void QSCUNode::yawCallback(const std_msgs::Float64::ConstPtr& msg){
	// Store the last command
	thruster_update = true;
	m_yaw_command_buffer = (float) msg->data;
}

void QSCUNode::pitchCallback(const std_msgs::Float64::ConstPtr& msg){
	thruster_update = true;
	m_pitch_command_buffer = (float) msg->data;
}

void QSCUNode::rollCallback(const std_msgs::Float64::ConstPtr& msg){
	thruster_update = true;
	m_roll_command_buffer = (float) msg->data;
}

void QSCUNode::depthCallback(const std_msgs::Float64::ConstPtr& msg){
	thruster_update = true;
	m_pitch_command_buffer = (float) msg->data;
}

void QSCUNode::surgeCallback(const std_msgs::Float64::ConstPtr& msg){
	thruster_update = true;
	m_surge_command_buffer = (float) msg->data;
}

void QSCUNode::swayCallback(const std_msgs::Float64::ConstPtr& msg){
	thruster_update = true;
	m_sway_command_buffer = (float) msg->data;
}
