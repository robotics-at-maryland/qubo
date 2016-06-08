
#include "dvl_tortuga.h"

DVLTortugaNode::DVLTortugaNode(std::shared_ptr<ros::NodeHandle> n, int rate, int board_fd, std::string board_file) : 
    SensorBoardTortugaNode(n, rate, board_fd, board_file){
	ros::Rate loop_rate(rate);
    //SG: TODO gotta make the name not hardcoded..
	publisher = n->advertise<underwater_sensor_msgs::DVL>("qubo/dvl", 1000);
}

DVLTortugaNode::~DVLTortugaNode(){}

void DVLTortugaNode::update(){
    publisher.publish(msg);
	ros::spinOnce();
}

void DVLTortugaNode::dvlCallBack(const underwater_sensor_msgs::DVL sim_msg){
	
}
