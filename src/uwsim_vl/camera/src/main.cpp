#include "camera_sim.h"
/**
   The basic idea here is to pass in a boolean from a launch script that determines if our class is 
   a real one or a simulated one. after that they behave exactly the same as far as main is concerned
**/

int main(int argc, char **argv){

    if(argc != 4){
        ROS_ERROR("The camera node received %i arguments which is not right\n (expected 4)", argc);
        exit(1);
    }
    
    std::shared_ptr<ros::NodeHandle> n;
    ros::init(argc, argv, "camera_node"); /** basically always needs to be called first */
 
    std::unique_ptr<RamNode> node;
     
    if(strcmp(argv[1], "simulated") == 0){
        node.reset(new CameraSimNode(n, 10)); /** 10 (the rate) is completely arbitrary */
    }else if(strcmp(argv[1], "tortuga") == 0) {
        // node.reset(new CameraTortugaNode(argc, argv, 10));
        //not currently written 
    }else{
        ROS_ERROR("the passed in arguments to camera node (%s) doesn't match anything that makes sense..\n", argv[1]); 
    }
    
    while (ros::ok()){
        node->update();
    }
    
}
