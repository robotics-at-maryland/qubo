#include "depth_controller.h"
#include <thread>

using namespace std;
using namespace ros;

int main(int argc, char* argv[]){
	
	init(argc, argv, "control_node");
	NodeHandle nh;

	DepthController node(nh);
	
	while(1){
		node.update();
		this_thread::sleep_for(chrono::seconds(1)); //this is really slow right now
	}
	
	return 0;
}
