#include "rotational_controller.h"

int main(int argc, char **argv) {
    ros::init(argc, argv, "rotational_controller");
    std::shared_ptr<ros::NodeHandle> n(new ros::NodeHandle);
    
    //TODO make work for tortuga node?

    RotationalController node(n, 10);
   
    while (ros::ok()){
        node.update();
    }
   
    return 0; 
}
