#include "pid_controller.h"
#include <thread>

using namespace std;
using namespace ros;



int main(int argc, char* argv[]){
	
	init(argc, argv, "depth_controller");
	NodeHandle nh;

		
	PIDController depth_node(nh, "depth");
	PIDController yaw_node(nh, "yaw");
	
	while(1){
		node.update();
		this_thread::sleep_for(chrono::seconds(1)); //this is really slow right now
	}
	
	return 0;
}
