//!  This is the abstract base class that all our vehicle nodes should inherit from

/*! 
 * The basic idea to have a real and a simulated version of every sensor, BOTH will inherit 
 * From this class, which means we call both the real and the simulated nodes in exactly the same manner.
 *
 * We have a publish and an update method that we insist all inherited classes use
 * We also go ahead and make a node handle and a publisher variable here. 
 */


#ifndef QUBONODE_HEADER
#define QUBONODE_HEADER

#include "ros/ros.h"
#include <iostream>
#include "std_msgs/String.h"
#include <sstream>


class QuboNode {
 public:
 
  QuboNode(){}; /**<Constructor, you should really never call this directly */
  ~QuboNode(){}; //Destructor 

 
  virtual void update() = 0;
  virtual void publish() = 0;
  //void sendAction(){};  // this isn't a pure function because sub classes won't necessarily use it.


  //We'll probably need a few more things 
 protected:
  ros::NodeHandle n; /**< the handle for the whole node */
  ros::Publisher publisher; /** simulated or real, we'll need a subscriber either way. */
  
};


#endif
