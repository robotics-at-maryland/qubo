#ifndef TEMP_TORTUGA_H
#define TEMP_TORTUGA_H

//#include <string>
//#include <random>
#include "tortuga_node.h" //always included
#include "tortuga/sensorapi.h"
#include "ram_msgs/Temp.h"

#define DEFAULT_TEMP 23.0

class TempTortugaNode : public TortugaNode {
    public: //public methods
        TempTortugaNode(int, char **, int); //constructor: first three fields mandatory, then specific fields to the node
        TempTortugaNode(int, char **, int, int, std::string);
        ~TempTortugaNode(); //destructor, necessary
  
        //update: retrieves data from any other node needed for operation.
        void update();

    protected: //fields 
        ram_msgs::Temp msg; //always include this, used to create specific message file for this node
		int fd;
        std::string sensor_file;
		ros::Publisher publishers[NUM_TEMP_SENSORS];
};

#endif
