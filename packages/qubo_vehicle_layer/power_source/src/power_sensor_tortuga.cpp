#include "power_sensor_tortuga.h"

PowerNodeTortuga::PowerNodeTortuga(int argc, char **argv, int inputRate, std::string node_name) {
    name = node_name;
    for ( int i = 0; i < 6; i ++ ) {
      std::string indiv_name = node_name + std::to_string(i);
      publisher[i] = n.advertise<ram_msgs::PowerSource>(("tortuga/power_source/" + indiv_name), 1000);
      msg[i].source_name = indiv_name;
      msg[i].enabled = DEFAULT_STATUS;
    }
}

PowerNodeTortuga::~PowerNodeTortuga() {}

void PowerNodeTortuga::update() {
  int v = readBatteryVoltages(sensor_fd, &info);
  if (checkError(v)) {
    // error occured, has been logged in checkError, not sure what should be done here
  }
  int i = readBatteryCurrents(sensor_fd, &info);
  if (checkError(i)) {
    // error occured, has been logged in checkError, not sure what should be done here
  }
  /* This doesn't return an error code, it bitmasks part of readStatus()
    Not sure which battery this actually is, so for now I'm just setting all them to this value
  */
  int life = readBatteryUsage(sensor_fd);
  for ( int i = 0; i < 6; i++ ) {
    msg[i].voltage = info.battVoltages[i];
    msg[i].current = info.battCurrents[i];
    msg[i].life = life;
  }
  ros::spinOnce();
}

void PowerNodeTortuga::publish() {
  for ( int i = 0; i < 6; i++ ) {
    publisher[i].publish(msg[i]);
  }
}
