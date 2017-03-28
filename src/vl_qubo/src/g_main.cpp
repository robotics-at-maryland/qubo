#include "g_control_node.h"

//includes for threading
#include <thread>

using namespace std; 

void nodeThread(int argc, char* argv[]){
        
    ros::init(argc, argv, "control_node");
    ros::NodeHandle nh;

    GControlNode cn(nh, "control_node", "/qubo/pose/");

    while(1){
        cn.update();
        sleep(.5);
    }
}



int main(int argc, char* argv[]){

    thread first(nodeThread, argc, argv);
    
    
    while(1){
        cout << "hello I'm the main thread" << endl;
        sleep(1);
    }
    
    return 0;
}
