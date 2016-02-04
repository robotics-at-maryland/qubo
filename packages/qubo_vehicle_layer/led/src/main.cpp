#include "led_sim.h"

int main(int argc, char **argv){
  ros::init(argc, argv, "led_node");
  bool simulated = true;

  LedSimNode *test = new LedSimNode(argc, argv, 10, "TEST");

  test->enable();

  while (ros::ok()) {
    test->publish();
    test->update();
  }
}
