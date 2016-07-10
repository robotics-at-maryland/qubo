#include "dvl_tortuga.h"

DVLTortugaNode::DVLTortugaNode(std::shared_ptr<ros::NodeHandle> n, int rate, int board_fd, std::string board_file):RamNode(n) {

    publisher = n->advertise<ram_msgs::DVL>("tortuga/dvl", 1000);
	ROS_DEBUG("Set up publisher");
    fd = board_fd;
    file = board_file;
}

DVLTortugaNode::~DVLTortugaNode(){}

void DVLTortugaNode::update(){
//	ros::spinOnce();

    checkError(readDVLData(fd, &raw));
	ROS_DEBUG("Read DVL Data");
    // Raw data has a pointer to complete packet
    pkt = raw.privDbgInf;
    if(raw.xvel_btm == DVL_BAD_DATA || raw.yvel_btm == DVL_BAD_DATA || raw.zvel_btm == DVL_BAD_DATA){
	ROS_ERROR("Bad Data");

    }
    
    else{
       
    	// Set all the message's data
    	msg.sysconf = pkt->sysconf;
    	msg.xvel_btm = raw.xvel_btm;
    	msg.yvel_btm = raw.yvel_btm;
    	msg.zvel_btm = raw.zvel_btm;
    	msg.evel_btm = pkt->evel_btm;
    	msg.beam1_range = raw.beam1_range;
    	msg.beam2_range = raw.beam2_range;
    	msg.beam3_range = raw.beam3_range;
    	msg.beam4_range = raw.beam4_range;
    	msg.btm_status = pkt->btm_status;
    	msg.xvel_ref = pkt->xvel_ref;
    	msg.yvel_ref = pkt->yvel_ref;
    	msg.zvel_ref = pkt->zvel_ref;
    	msg.evel_ref = pkt->evel_ref;
    	msg.ref_lyr_start = pkt->ref_lyr_start;
    	msg.ref_lyr_end = pkt->ref_lyr_end;
    	msg.ref_lyr_status = pkt->ref_lyr_status;
    	msg.TOFP_hour = pkt->TOFP_hour;
    	msg.TOFP_min = pkt->TOFP_min;
    	msg.TOFP_sec = pkt->TOFP_sec;
    	msg.TOFP_hnd = raw.TOFP_hundreths;
    	msg.BIT_results = pkt->BIT_results;
    	msg.speed_of_sound = pkt->speed_of_sound;
    	msg.temperature = pkt->temperature;
    	msg.checksum = pkt->checksum;

    	publisher.publish(msg);

	ROS_DEBUG("Published msg");

    }
}

bool DVLTortugaNode::checkError(int e) {
    switch(e) {
    case ERR_NOSYNC:
      ROS_ERROR("NOSYNC ERROR in node %s", file.c_str());
      return true;
    case ERR_TOOBIG:
      ROS_ERROR("TOOBIG ERROR in node %s", file.c_str());
      return true;
    case ERR_CHKSUM:
      ROS_ERROR("CHKSUM ERROR in node %s", file.c_str());
      return true;
    default:
      return false;
    }
}


