#ifndef SENSOR_BOARD_TORTUGA_H
#define SENSOR_BOARD_TORTUGA_H

<<<<<<< HEAD:packages/qubo_vehicle_layer/sensor_board/include/sensor_board_tortuga.h
#include "thruster_tortuga.h"
#include "temp_tortuga.h"
=======
#include "tortuga_node.h"
#include "sensorapi.h"

/*
 * Dependencies for thrusters  
 */
#include "std_msgs/Int64MultiArray.h"
#include "sensor_msgs/Joy.h"

/*
 * Dependencies for temperature
 */
#include "std_msgs/UInt8MultiArray.h"
#include "ram_msgs/Temperature.h"
>>>>>>> vehicle_refactor_2016:packages/vehicle_layer/sensor_board/include/sensor_board_tortuga.h

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
