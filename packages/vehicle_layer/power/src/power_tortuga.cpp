#include "power_tortuga.h"

PowerNodeTortuga::PowerNodeTortuga(std::shared_ptr<ros::NodeHandle> n, int rate, int board_fd, std::string board_file):
  SensorBoardTortugaNode(n, rate, board_fd, board_file) {
    for ( int i = 0; i < 6; i ++ ) {
      std::string pub_name = "tortuga/power_source" + std::to_string(i);
      publisher[i] = n->advertise<ram_msgs::PowerSource>(pub_name, 1000);
    }
}

PowerNodeTortuga::~PowerNodeTortuga() {}

void PowerNodeTortuga::update() {
	checkError(readBatteryVoltages(fd, &info));
	checkError(readBatteryCurrents(fd, &info));

	/* This doesn't return an error code, it bitmasks part of readStatus()
	Not sure which battery this actually is, so for now I'm just setting all them to this value
  */
	int life = readBatteryUsage(fd);
	for ( int i = 0; i < 6; i++ ) {
		msg.voltage = info.battVoltages[i];
		msg.current = info.battCurrents[i];
		msg.life = life;
		publisher[i].publish(msg);
	}
	ros::spinOnce();

}
