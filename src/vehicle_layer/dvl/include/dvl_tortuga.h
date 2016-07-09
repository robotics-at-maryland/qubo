#ifndef DVL_TORTUGA_HEADER
#define DVL_TORTUGA_HEADER

#include "ram_node.h"
#include "ram_msgs/DVL.h"
#include "dvlapi.h"

#define DVL_BAD_DATA -32768

class DVLTortugaNode : public RamNode {
  public:
    DVLTortugaNode(std::shared_ptr<ros::NodeHandle>, int rate, int fd ,  std::string file_name);
    ~DVLTortugaNode();

	void update();
	bool checkError(int e);


protected:
	int fd;
	std::string file;
	ram_msgs::DVL msg;
	ros::Publisher publisher;
  // What the dvl api gives
	RawDVLData raw;
	CompleteDVLPacket *pkt;
};

#endif
