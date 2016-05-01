#ifndef BATTSIM_HEADER
#define BATTSIM_HEADER

#include "tortuga_node.h"
#include "ram_msgs/PowerSource.h"

#define DEFAULT_STATUS false
#define DEFAULT_VOLTAGE 24.0
#define DEFAULT_CURRENT 9001.0
#define DEFAULT_LIFE 3600

class PowerNodeTortuga : public TortugaNode {
    protected:
  /* Tortuga has 6 Batteries, so this node will have 6 publishers and 6 messages
     instead of the single publisher and message
   */
        ros::Publisher publisher[6];
        ram_msgs::PowerSource msg[6];

        struct powerInfo info;

    public:
        PowerNodeTortuga(int, char **, int, std::string);
        ~PowerNodeTortuga();

        void update();
        void publish();
};

#endif
