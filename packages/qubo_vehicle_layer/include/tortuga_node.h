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

#define IMU_FD "t/imu_fd"
#define SENSOR_FD "t/sensor_fd"


#include "qubo_node.h"
#include "tortuga/sensorapi.h"
#include "tortuga/imuapi.h"

class TortugaNode : QuboNode {
 public:

  TortugaNode(){}; /**<Constructor, you should really never call this directly */
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
      ROS_DEBUG("ERROR in node %s", name.c_str());

    }
  }
  

  //We'll probably need a few more things
 protected:
  // Node's name, used for debugging in checkerror
  std::string name;
  const int sensor_fd;
  const int imu_fd;
};




#endif
