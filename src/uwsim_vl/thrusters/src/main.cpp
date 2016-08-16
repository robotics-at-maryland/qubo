#include "thruster_sim.h"

int main(int argc, char **argv){
    
    //check if the args are right, assumes we get called from a roslaunch script
    if(argc != 4){
        ROS_ERROR("The thruster node received %i arguments which is not right\n", argc);
        for(int i = 0; i < argc; i++){
            ROS_ERROR("arg: %s\n", argv[i]);
        }
        exit(1);
    }
    
    //initialize the ros node we'll use for the thrusters 
    std::shared_ptr<ros::NodeHandle> n;
    ros::init(argc, argv, "thruster_node"); 

    //pointer the node
    std::unique_ptr<RamNode> node;

    //check if we want to be simulated or real   
    if(strcmp(argv[1], "simulated") == 0){
        //set the pointer to and instance of a simulated node
        node.reset(new ThrusterSimNode(n,10)); /** 10 (the rate) is completely arbitrary */
    }else if(strcmp(argv[1], "tortuga") == 0) {
        ROS_ERROR("you're trying to launch tortugas thrusters from the thruster main, but you need to do it through the sensor board!");
    }else{
        ROS_ERROR("the passed in arguments to thruster node (%s) doesn't match anything that makes sense..\n", argv[1]); 
        exit(1);
    }

    //main loop for the program
    while (ros::ok()){
        node->update();
    }
        
}
    
