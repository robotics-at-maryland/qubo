#include "dvl_sim.h"

/* this is the main method for dvl node in the uwsim vehicle layer, it's not currently implemented but I think we're
   moving towards using gazebo anyway so it may remain a TODO forever
*/

int main(int argc, char **argv){

  // initialize ros node for the DVL
	ros::init(argc, argv, "dvl_node");
    std::shared_ptr<ros::NodeHandle> n(new ros::NodeHandle);
    
    //TODO


    while (ros::ok()) {
      
	ros::Duration(.1).sleep();
	ROS_DEBUG("you'e calling a node that hasn't been implemented yet!");
    }
}
