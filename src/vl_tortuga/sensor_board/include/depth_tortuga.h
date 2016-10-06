//! This is tortugas version of the depth sensor node.
#ifndef DEPTHT_HEADER //I don't really see anybody needing to inherit this class, but better safe than sorry.
#define DEPTHT_HEADER

#include "sensor_board_tortuga.h"
#include "sensorapi.h"
#include "std_msgs/Float32.h"

class DepthTortugaNode : public SensorBoardTortugaNode {
	
    public:
    DepthTortugaNode(std::shared_ptr<ros::NodeHandle> n, int rate, int fd, std::string file_name);
    ~DepthTortugaNode();

	void update();
	void depthCallBack(const std_msgs::Float32 msg);
	
	protected:
	std_msgs::Float32 msg;
	ros::Publisher publisher;
    int fd;
  std::string sensor_file;
	//int fd; //the file descriptor, established by the a call to openSensorBoard
	
};

#endif
