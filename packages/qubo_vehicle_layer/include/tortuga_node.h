//!  This is the abstract base class that all our vehicle nodes should inherit from

/*! 
 * Tortuga node is a subclass of Qubo Node that will do the same thing but includes
   a couple of extra things specific to Tortuga
 */


#ifndef QUBONODE_TORTUGA_HEADER
#define QUBONODE_TORTUGA_HEADER

// This is the file name that the fd describes
#define IMU_BOARD "imu_file_descriptor"
#define SENSOR_BOARD "sensor_board_file_descriptor"
#include "qubo_node.h"
#include "tortuga/sensorapi.h"
#include "tortuga/imuapi.h"

class TortugaNode {
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
      return true;
    default:
      return false;
    }
  }
  

  //We'll probably need a few more things 
 protected:
  std::string name;
  static const int fd_sens;
  static const int fd_imu;
  ros::NodeHandle n;
  /*
  //this code can't be here 
  if(!n.getParam(IMU_BOARD, fd_imu)){
    ROS_ERROR("IMU file descriptor not found on param server");
  };
  if(!n.getParam(SENSOR_BOARD, fd_sens)){
    ROS_ERROR("Sensor board file descriptor not found on param server");
  };
  */
};




#endif
