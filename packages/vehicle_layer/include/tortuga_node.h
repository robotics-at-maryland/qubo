
/*!
 * Tortuga node is a subclass of Qubo Node that will do the same thing but includes
 a couple of extra things specific to Tortuga
*/


#ifndef QUBONODE_TORTUGA_HEADER
#define QUBONODE_TORTUGA_HEADER


#include "qubo_node.h"
#include "sensorapi.h"
#include "imuapi.h"

class TortugaNode : public QuboNode {
    protected:
    // Node's name, used for debugging in checkerror
    std::string name;
    int sensor_fd = 0;
    int imu_fd = 0;
    public:

    // ISSUE: might be because n is defined in QuboNode namespace and needs the variables to set to
    // be defined in same place
    /**<Constructor, you should really never call this directly */
    TortugaNode(){};
    ~TortugaNode(){}; //Destructor

    /*
      Error codes are defined in sensorapi.h
      Use this function to log check if an error has occured using sensorapi
      Returns if an error happened and logs it
    */


    bool checkError(int e) {
        switch(e) {
        case SB_IOERROR:
            ROS_DEBUG("IO ERROR in node %s", name.c_str());
            return true;
        case SB_BADCC:
            ROS_DEBUG("BAD CC ERROR in node %s", name.c_str());
            return true;
        case SB_HWFAIL:
            ROS_DEBUG("HW FAILURE ERROR in node %s", name.c_str());
            return true;
        case SB_ERROR:
            ROS_DEBUG("SB ERROR in node %s", name.c_str());
            
        }
    }

};

#endif
