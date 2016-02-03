#ifndef TEMPSIM_HEADER
#define TEMPSIM_HEADER

#include <string>
#include "qubo_node.h"
#include "ram_msgs/Temperature.h"

#define DEFAULT_TEMP 23.0

class TempSimNode : QuboNode {
    protected:
        std::string sensorName;
        ram_msgs::Temperature msg;
        double temp;

    public:
        TempSimNode(int, char **, int, std::string);
        ~TempSimNode();

        void update();
        void publish();
};

#endif
