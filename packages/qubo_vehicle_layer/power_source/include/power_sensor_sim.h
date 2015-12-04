#ifndef BATTSIM_HEADER
#define BATTSIM_HEADER

#include <string>
#include "qubo_node.h"
#include "ram_msgs/PowerSource.h"

class PowerSimNode : QuboNode {
    public:
        PowerSimNode(int, char **, int, std::string);
        ~PowerSimNode();

        void update();
        void publish();

    protected:
        ram_msgs::PowerSource msg;
        bool enabled;
        float voltage;
        float current;
};

#endif
