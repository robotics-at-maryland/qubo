#include "dvl_sim.h"
#include "dvl_tortuga.h"

int main(int argc, char **argv){

  // initialize ros node for the DVL
	ros::init(argc, argv, "dvl_node");
    std::shared_ptr<ros::NodeHandle> n(new ros::NodeHandle);
    
  // open the DVL
    ROS_ERROR("Starting to open file");
   std::string dvl_file = "/dev/dvl";
    int fd = openDVL(dvl_file.c_str());
	ROS_ERROR("Opened file, fd = %i", fd);

    std::unique_ptr<DVLTortugaNode> dvl_node;
	ROS_ERROR("Initialized pointer");

    if(strcmp(argv[1], "simulated") == 0) {
      //TODO
    } else if (strcmp(argv[1], "tortuga") == 0) {
	  ROS_ERROR("In tortuga cmp");
      dvl_node.reset(new DVLTortugaNode(n, 10, fd, dvl_file));
	  ROS_ERROR("Set pointer");

    } else {
      ROS_ERROR("the pased in arguments to sensor board node (%s) doesn't match anything that makes sense...", argv[1]);
      exit(1);
    }

    while (ros::ok()) {
      
	ros::Duration(.5).sleep();
	ROS_DEBUG("DVLMAIN: calling update");
	dvl_node->update();
    }
}
