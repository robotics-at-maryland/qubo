#include "yaw_controller.h"
#include <thread>

using namespace std;
using namespace ros;

int main(int argc, char* argv[]){
	
	init(argc, argv, "yaw_controller");
	NodeHandle nh;

	YawController node(nh);
	
	while(1){
		node.update();
		this_thread::sleep_for(chrono::seconds(1)); //this is really slow right now
	}
	
	return 0;
}
