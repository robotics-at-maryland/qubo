#include "dvl_sim.h"
#include "dvl_tortuga.h"

int main(int argc, char **argv){

	if(argc != 4){
        ROS_ERROR("The DVL node received %i arguments which is not right\n", argc);
        exit(1);
    }

    std::shared_ptr<ros::NodeHandle> n;
    ros::init(argc, argv, "dvl_node"); /** basically always needs to be called first */

    
    std::unique_ptr<RamNode> node;
     
    if(strcmp(argv[1], "simulated") == 0){
        node.reset(new DVLSimNode(n, 10)); /** 10 (the rate) is completely arbitrary */
    }else if(strcmp(argv[1], "tortuga") == 0) {
        ROS_ERROR("you tried to launch the torgua dvl node independently but you need to do so through the sensor board now");
        exit(1);
    }else{
        ROS_ERROR("the passed in arguments to DVL node (%s) doesn't match anything that makes sense..\n", argv[1]); 
    }
       
    while (ros::ok()){
        node->update();
    }

}
