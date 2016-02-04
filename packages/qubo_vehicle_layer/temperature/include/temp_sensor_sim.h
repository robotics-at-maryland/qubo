#ifndef TEMPSIM_HEADER
#define TEMPSIM_HEADER

#include <string>
#include <random>
#include "qubo_node.h"
#include "ram_msgs/Temperature.h"

#define DEFAULT_TEMP 23.0

class TempSimNode : QuboNode {
    protected:
        std::string sensorName;
        ram_msgs::Temperature msg;
        double real_temp;
        double lower_bound = -1;
        double upper_bound = 1;
//       std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
//        std::default_random_engine re;

    public:
        TempSimNode(int, char **, int, std::string);
        ~TempSimNode();

        void update();
        void publish();
};

#endif
