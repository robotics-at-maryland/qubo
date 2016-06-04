#include "sensor_board_tortuga.h"

#include "thruster_tortuga.h"
//include your header here, no need to relocate it.

//Sean wrote most of this, direct questions to him

/**
    This is the main method for the sensor_board node,
    it will launch use a lot of different classes for the different
    components but feed them all the same node handle so
    that they all run in the same process. This makes it possible
    for them to all communicate with the sensor board directly, 
    which makes our life a lot easier. 
**/

int main(int argc, char **argv) {

    //initialize the ros node we'll use for anything wanting to talk to the sensor board. 
    std::shared_ptr<ros::NodeHandle> n;
    ros::init(argc, argv, "sensor_board_node");

    //open the sensor board
    std::string sensor_file = "/dev/sensor";
    int fd = openIMU(sensor_file.c_str());


    //we don't know what type of node we want until we look at the input arguments. 
    std::unique_ptr<SensorBoardTortugaNode> thruster_node;
    //SG: add a unique_ptr to your node as well

    
    //TODO:
    //SG: This part may have to be a little different for this node, we may want to read from a config file, or just pass in more arguments
    //to specify which sensors are real vs simlated, for now we'll just make sure it works in the case where everything is the real tortuga version
    
    if(strcmp(argv[1], "simulated") == 0) {
        //TODO
    } else if (strcmp(argv[1], "tortuga") == 0) {
        thruster_node.reset(new ThrusterTortugaNode(n, 10, fd, sensor_file));

        //copy the above with your node, just make sure n, fd and sensor_file are the same, not sure if we need rate honestly and I'd like to remove it if possible

    } else {
        ROS_ERROR("the pased in arguments to sensor board node (%s) doesn't match anything that makes sense...", argv[1]);
        exit(1);
    }
    
    while (ros::ok()) {
        thruster_node->update();
        //make sure you run your nodes update here.
    }
}
