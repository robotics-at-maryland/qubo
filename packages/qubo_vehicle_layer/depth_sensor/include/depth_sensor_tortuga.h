//! This is tortugas version of the depth sensor node.
#ifndef DEPTHT_HEADER //I don't really see anybody needing to inherit this class, but better safe than sorry.
#define DEPTHT_HEADER

#include "tortuga_node.h"
#include "underwater_sensor_msgs/Pressure.h"

class DepthTortugaNode : public TortugaNode {

public:
  DepthTortugaNode(int, char**, int, std::string);
  ~DepthTortugaNode();

  void update();
  void publish();
  void depthCallBack(const underwater_sensor_msgs::Pressure msg);

 protected:
  underwater_sensor_msgs::Pressure msg;
  //int fd; //the file descriptor, established by the a call to openSensorBoard


};

#endif
