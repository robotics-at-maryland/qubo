#ifndef SENSOR_BOARD_TORTUGA_H
#define SENSOR_BOARD_TORTUGA_H

#include "thruster_tortuga.h"
#include "temp_tortuga.h"


class SensorBoardTortugaNode : public TortugaNode {
	public:
        SensorBoardTortugaNode(int, char**, int);
        ~SensorBoardTortugaNode();
        
        void update();
        

    protected:
    int fd; //File descriptor for sensor board
    std::string sensor_file; // Serial port/system location for sensor board

    std::unique_ptr<QuboNode> thrusters;
    std::unique_ptr<QuboNode> temperature;
};

#endif
