#include "thruster_sim.h"
#include "thruster_tortuga.h"

/**
   This is the main method for our depth_sensor node
**/


int main(int argc, char **argv){
    
    printf("got here0\n");

    //SG: argc may need to be different idk 
    /*  if(argc != 4){
        ROS_ERROR("The thruster node received %i arguments which is not right\n", argc);
        for(int i = 0; i < argc; i++){
            ROS_ERROR("arg: %s\n", argv[i]);
        }
        exit(1);
    }
    */

    ros::init(argc, argv, "thruster_node"); /** basically always needs to be called first */
  
    /**
       The basic idea here is to pass in a boolean from a launch script that determines if our class is 
       a real one or a simulated one. after that they behave exactly the same as far as main is concerned
       right now though we don't have a real depth sensor, hence the comments
    **/

    
    std::unique_ptr<QuboNode> node;
   
    if(strcmp(argv[1], "simulated") == 0){
        node.reset(new ThrusterSimNode(argc, argv, 10)); /** 10 (the rate) is completely arbitrary */
    }else if(strcmp(argv[1], "tortuga") == 0) {
        node.reset(new ThrusterTortugaNode(argc, argv, 10));
    }else{
        ROS_ERROR("the passed in arguments to thruster node (%s) doesn't match anything that makes sense..\n", argv[1]); 
        exit(1);
    }

    while (ros::ok()){
        node->update();
        node->publish();
        
    }
        
}
    
