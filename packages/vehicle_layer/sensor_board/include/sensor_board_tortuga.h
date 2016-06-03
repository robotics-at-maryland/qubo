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

    //SG: We don't want to do it this way, I want the sensor board to be one monolithic process,
    //spawning more nodes puts us back where we were before
    std::unique_ptr<QuboNode> thrusters;
    std::unique_ptr<QuboNode> temperature;


    //SG: should we try and handle some of there errors here?
    //we should look into how sensor api handles some of these,
    //for example on some errors we may want to panic immediately
    //and quit/shutdown.
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
            ROS_DEBUG("SB ERROR in node %s", name.c_str());

        }
    }
};

#endif
