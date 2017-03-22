#include "g_control_node.h"

//includes for threading
#include <thread>

using namespace std; 

void nodeThread(int argc, char* argv[]){
        
    ros::init(argc, argv, "control_node");
    ros::NodeHandle nh;

    GControlNode cn(nh, "control_node", "/basic_qubo/pose_gt");

    while(1){
        cn.update();
        sleep(.5);
    }
}


void thrusterThread(){
    //for the gazebo version of this thread we need to subscribe to the pitch/roll/yaw commands and translate them
    
    while(1){
        cout << "hey what's up I'm the thruster thread" << endl;
        sleep(1);
    }
    
}



int main(int argc, char* argv[]){

    thread first(nodeThread, argc, argv);
    thread second(thrusterThread);
    
    
    while(1){
        cout << "hello I'm the main thread" << endl;
        sleep(1);
    }
    
    return 0;
}
