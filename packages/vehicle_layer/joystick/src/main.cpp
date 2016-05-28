#include "JoyReader.h"

/**
   This is the main method for our depth_sensor node
**/


int main(int argc, char **argv){

    ros::init(argc, argv, "joy_node");
	std::unique_ptr<QuboNode> node;
    
   // if(strcmp(argv[1], "tortuga") == 0) {
        node.reset(new JoyReader(argc, argv, 10));
   // }else{
   // }


    while (ros::ok()){
        node->update();
        node->publish();
    }

}

