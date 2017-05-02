#include "gazebo_hardware_node.h"

//includes for threading
#include <thread>

using namespace std; 

// void nodeThread(int argc, char* argv[]){
        
//     ros::init(argc, argv, "control_node");
//     ros::NodeHandle nh;

//     GControlNode cn(nh, "control_node", "/qubo/pose/");

//     while(1){
//         cn.update();
//         sleep(.5);
//     }
// }

 
//I might thread this eventually to be a better simulation of what will really happen
//ros makes that hard though. If you want to do it you'll need to manually get the global callback
//queue
int main(int argc, char* argv[]){
	
	//    thread first(nodeThread, argc, argv);
    
	
    ros::init(argc, argv, "hardware_node");
    ros::NodeHandle nh;
	
	GazeboHardwareNode cn(nh, "hardware_node", "/qubo/pose/");
	
    while(1){
        cn.update();
        this_thread::sleep_for(chrono::seconds(1)); //this is really slow right now
    }
	
	return 0;
}
