//!  This is the abstract base class that all our vehicle nodes should inherit from

/*!
 * Tortuga node is a subclass of Qubo Node that will do the same thing but includes
 a couple of extra things specific to Tortuga
*/


#ifndef QUBONODE_TORTUGA_HEADER
#define QUBONODE_TORTUGA_HEADER

/* These define the parameter server names that will hold the value of the fd's
   Ideally these values are constant and each node should have access to it, so it would be
   best to have a way for the values to be saved in one place where every subclass of TortugaNode
   has access to it.
   For now every node has access to the parameter server's variable name IMU_BOARD and SENSOR_BOARD
   so each node can get the value.
*/

#define IMU_FD "imu_file_descriptor"
#define SENSOR_FD "sensor_board_file_descriptor"


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
    TortugaNode(){
        n.getParam(IMU_FD, imu_fd);
        n.getParam(SENSOR_FD, sensor_fd);
    };
    ~TortugaNode(){}; //Destructor

    /*
      Error codes are defined in sensorapi.h
      Use this function to log check if an error has occured using sensorapi
      Returns if an error happened and logs it
    */

    //SG: should we try and handle some of there errors here?
    //we should look into how sensor api handles some of these,
    //for example on some errors we may want to panic immediately 
    //and quit/shutdown.
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

    //We'll probably need a few more things
};

#endif
