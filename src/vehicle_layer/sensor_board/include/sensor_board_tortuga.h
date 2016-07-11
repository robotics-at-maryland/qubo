#ifndef SENSOR_BOARD_TORTUGA_H
#define SENSOR_BOARD_TORTUGA_H

#include "ram_node.h"
#include "sensorapi.h"


class SensorBoardTortugaNode : public RamNode {
    public:
    SensorBoardTortugaNode(std::shared_ptr<ros::NodeHandle>, int , int, std::string);
    ~SensorBoardTortugaNode();
   

        void update();
        

    protected:
    int fd; //File descriptor for sensor board
    std::string sensor_file; // Serial port/system location for sensor board


    //SG: should we try and handle some of there errors here?
    //we should look into how sensor api handles some of these,
    //for example on some errors we may want to panic immediately
    //and quit/shutdown.
    bool checkError(int);
};

#endif
