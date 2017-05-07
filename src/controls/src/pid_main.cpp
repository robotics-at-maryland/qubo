#include "pid_controller.h"
#include <thread>

using namespace std;
using namespace ros;



int main(int argc, char* argv[]){
	
	//this just parses arguments, should handle being launched from rosrun or roslaunch 
	if(argc < 3){
		ROS_ERROR("less than expected number of arguments\n"
				  "we need you to pass in the topic you want to control\n"
				  "usage from rosrun - rosrun controls pid_controller <topic_name> <update_rate(Hz)>"
				  "to use from ros launch  - <node name=\"qubo_controller\" pkg=\"controls\" type=\"pid_controller\" args=\"depth 100\" />");

		
		ROS_ERROR("for reference the arguments you passed in are:");
		for(int i = 0; i < argc; i++){
			ROS_ERROR("%s", argv[i]);
		}

		exit(0);
	}

	//first argument is the executable, second should be our name... roslaunch has the potential
	//to mess with this
	ROS_ERROR("the topic your controller is going to use to publish to is %s and it will publish at %s, please make sure that is correct", argv[1], argv[2]);
	string control_topic = argv[1];
	string freq = argv[2]; //argv[2] needs to be the desired freq in Hz
		
	
	init(argc, argv, control_topic + "_" + "controller");
	NodeHandle nh;
	
	PIDController node(nh, control_topic);

	Rate rate(stoi(freq)); //rate object to sleep the right amount of time between calls to update
	
	
	
	
	while(1){
		node.update();
		rate.sleep(); //this will sleep at a rate 1/argv[2]
	}
	
	return 0;
}
