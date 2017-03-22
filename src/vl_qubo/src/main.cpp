#include "control_node.h"

//includes for threading
#include <thread>

using namespace std;

void nodeThread(int argc, char* argv[]){
    
    cout << argc << endl;
    cout << argv[0] << endl;
    
    ros::init(argc, argv, "control_node");
    ros::NodeHandle nh;

    //    ControlNode cn(nh, "control_node", "/dev/something","/dev/something");

    //    ros::spin();
    while(1){
        cout << "hello I'm the node thread" << endl;
        sleep(.5);
    }
    
}

void tivaThread(){
    while(1){
        cout << "hello I'm the tiva thread" << endl;
        sleep(1);
    }
}


int main(int argc, char* argv[]){

    thread first(nodeThread, argc, argv);
    thread second(tivaThread);

    // ros::spin();

    while(1){
        cout << "hello I'm the main thread" << endl;
        sleep(1);
    }
    
    return 0;
}

