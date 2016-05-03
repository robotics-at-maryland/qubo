#ifndef DVL_TORTUGA_HEADER
#define DVL_TORTUGA_HEADER

#include "tortuga_node.h"
#include "underwater_sensor_msgs/DVL.h"

class DVLTortugaNode : public TortugaNode {
public:
	DVLTortugaNode(int, char**, int);
	~DVLTortugaNode();

	void update();
	void dvlCallBack(const underwater_sensor_msgs::DVL msg);

protected:
	underwater_sensor_msgs::DVL msg;
	ros::Subscriber subscriber;
};

#endif
