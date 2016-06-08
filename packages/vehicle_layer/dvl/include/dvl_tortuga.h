#ifndef DVL_TORTUGA_HEADER
#define DVL_TORTUGA_HEADER

#include "sensor_board_tortuga.h"
#include "underwater_sensor_msgs/DVL.h"

class DVLTortugaNode : public SensorBoardTortugaNode {
    public:
	DVLTortugaNode(std::shared_ptr<ros::NodeHandle> n, int rate , int fd, std::string board_file);
	~DVLTortugaNode();
    
	void update();
	void dvlCallBack(const underwater_sensor_msgs::DVL msg);
    
    protected:
	underwater_sensor_msgs::DVL msg;
	ros::Subscriber subscriber;
	ros::Publisher publisher;
};

#endif
