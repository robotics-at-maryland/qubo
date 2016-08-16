#include "translational_controller.h"

int main(int argc, char **argv) {
    ros::init(argc, argv, "translational_controller");
    std::shared_ptr<ros::NodeHandle> n(new ros::NodeHandle);
    
    //TODO make work for tortuga node?
    TranslationalController node(n,10);
    while (ros::ok()){
        node.update();
    }

    return 0;    
}
