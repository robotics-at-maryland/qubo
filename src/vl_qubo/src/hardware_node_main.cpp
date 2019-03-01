#include "hardware_node.h"

#include <thread>

using namespace std;

int main(int argc, char* argv[]) {
  ros::init(argc, argv, "hardware_node");
  ros::NodeHandle nh;

  HardwareNode cn(nh, "hardware_node");

  while(1) {
    cn.update();
    this_thread::sleep_for(chrono::seconds(1));
  }

  return 0;

}
