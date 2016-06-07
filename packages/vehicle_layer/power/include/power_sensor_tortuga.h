#ifndef BATT_TORTUGA_H
#define BATT_TORTUGA_H

#include "sensor_board_tortuga.h"
#include "sensorapi.h"
#include "ram_msgs/PowerSource.h"

#define DEFAULT_STATUS false
#define DEFAULT_VOLTAGE 24.0
#define DEFAULT_CURRENT 9001.0
#define DEFAULT_LIFE 3600

class PowerNodeTortuga : public SensorBoardTortugaNode {

    public:
      PowerNodeTortuga(std::shared_ptr<ros::NodeHandle>, int rate, int fd ,  std::string file_name);
      ~PowerNodeTortuga();

      void update();

    protected:
  /* Tortuga has 6 Batteries, so this node will have 6 publishers and 6 messages
     instead of the single publisher and message
   */
      int fd;
      ros::Publisher publisher[6];
      ram_msgs::PowerSource msg;
      struct powerInfo info;
};

#endif
