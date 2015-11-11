#ifndef QUBONODE_HEADER
#define QUBONODE_HEADER

#include "ros/ros.h"
#include <iostream>
#include "std_msgs/String.h"
#include <sstream>


class QuboNode {
 public:
  QuboNode(){}; //Constructor, you should really never call this directly
  ~QuboNode(){}; //Destructor 

 
  virtual void subscribe() = 0;
  virtual void publish() = 0;
  //void sendAction(){};  //this isn't a pure function because sub classes won't necessarily use it. 


  //We'll probably need a few more things 
 protected:
  ros::NodeHandle n; //the handle for the whole node
  ros::Publisher publisher; //simulated or real, we'll need a subscriber either way. 
  //ros::Rate rate;
  
};


#endif
