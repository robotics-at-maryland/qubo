#include "joy_reader.h"

/**
   This is the main method for our depth_sensor node
**/

//SG: no need for that pointer reset magic we did for the vehicle layer
int main(int argc, char **argv){

    ros::init(argc, argv, "joy_reader_node");    
    JoyReader node(argc, argv, 10);
    
    
    while (ros::ok()){
	ROS_ERROR("Updating");
        node.update();    
    }

}
    
