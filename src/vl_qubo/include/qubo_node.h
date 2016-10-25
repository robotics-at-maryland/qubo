//!  This is the abstract base class that all our vehicle nodes should inherit from

/*!
 * The basic idea to have a real and a simulated version of every sensor, BOTH will inherit
 * From this class, which means we call both the real and the simulated nodes in exactly the same manner.
 *
 * We have a publish and an update method that we insist all inherited classes use
 * We also go ahead and make a node handle and a publisher variable here.
 *
 * The REASON we insist everything implement an update method rather than just using ros::spin or something like that
 * is so we can update the real sensors correctly.
 */


#ifndef QUBONODE_HEADER
#define QUBONODE_HEADER

#include "ros/ros.h"
#include <iostream>
#include "std_msgs/String.h"
#include <sstream>
#include <thread>

class QuboNode {
public:

    /**<Constructor, you should really never call this directly, all it does is set our version of the node handle the the passed in node handle. */
    QuboNode(std::shared_ptr<ros::NodeHandle> n ){
        this->n = n;
    };

    ~QuboNode(){}; //Destructor


    virtual void update() = 0;

    void sleep() {
        loop_rate->sleep();
    }

protected:
    std::shared_ptr<ros::NodeHandle> n; /**< the handle for the whole node */
    std::unique_ptr<ros::Rate> loop_rate;

};


#endif
