#ifndef TEMP_TORTUGA_H
#define TEMP_TORTUGA_H

//#include <string>
//#include <random>
#include "sensorapi.h"
#include "sensor_board_tortuga.h"
//#include "ram_msgs/Temp.h"
#include "std_msgs/Char.h"


class TempTortugaNode : public SensorBoardTortugaNode {

    public: //public methods

        TempTortugaNode(std::shared_ptr<ros::NodeHandle>, int rate, int fd, std::string file_name);
        ~TempTortugaNode();

        //update: retrieves data from any other node needed for operation.
        void update();

    protected: //fields 
        std_msgs::Char msg; //always include this, used to create specific message file for this node
        int fd;
        std::string sensor_file;
        ros::Publisher publishers[NUM_TEMP_SENSORS];
};

#endif
