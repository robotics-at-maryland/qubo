#ifndef BATTSIM_HEADER
#define BATTSIM_HEADER

#include <string>
#include "qubo_node.h"
#include "ram_msgs/PowerSource.h"

#define DEFAULT_STATUS true
#define DEFAULT_VOLTAGE 24.0
#define DEFAULT_CURRENT 9001.0
#define DEFAULT_LIFE 3600

class PowerSimNode : QuboNode {
    protected:
        static std::string currentSource;
        ram_msgs::PowerSource msg;
        std::string sourceName;
        bool enabled;
        float voltage;
        float current;
        int life;

        double voltageDrainRate = DEFAULT_VOLTAGE / DEFAULT_LIFE;
        double currentDrainRate = DEFAULT_CURRENT / DEFAULT_LIFE;
        std::time_t startTime;
    
    public:
        PowerSimNode(int, char **, int, std::string);
        ~PowerSimNode();

        void update();
        void publish();
        static void setCurrentSource(std::string);
};

#endif
