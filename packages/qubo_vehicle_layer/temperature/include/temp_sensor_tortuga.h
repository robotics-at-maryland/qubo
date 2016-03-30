#ifndef TEMPSIM_HEADER
#define TEMPSIM_HEADER

//#include <string>
//#include <random>
#include "qubo_node.h" //always included
#include "std_msgs/UInt8MultiArray"
#include "sensorapi.h"

#define DEFAULT_TEMP 23.0

class TempSimNode : public QuboNode {
    public: //public methods
        TempSimNode(int, char **, int, std::string); //constructor: first three fields mandatory, then specific fields to the node
        ~TempSimNode(); //destructor, necessary
  
        //update: retrieves data from any other node needed for operation.
        void update();
        //publish: puts data about the node in the message file.
        void publish();

    protected: //fields 
        std_msgs::UInt8MultiArray msg; //always include this, used to create specific message file for this node
		int fd;
		ros::Subscriber subscriber;
};

#endif
