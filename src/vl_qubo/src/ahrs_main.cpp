#include "ahrs_node.h"

//includes for threading
#include <thread>

//include for exiting
#include <csignal>

using namespace std;

void signal_handler(int sig) {
  exit(SIGINT);
}



int main(int argc, char* argv[]){
	ros::init(argc, argv, "ahrs_node");
    ros::NodeHandle nh;
	
	AHRSQuboNode cn(nh, "hardware_node", "/dev/ttyUSB0");
	
    while(1){
        cn.update();
        this_thread::sleep_for(chrono::seconds(1)); //this is really slow right now
    }
	
	return 0;
}
