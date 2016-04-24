#include "led_sim.h"

//main is what creates and runs individual nodes
int main(int argc, char **argv){
  ros::init(argc, argv, "led_node"); //initiate node (this name goes in makefile)
  bool simulated = true;

  LedSimNode *test = new LedSimNode(argc, argv, 10, "TEST"); //create instance of a node

  test->enable();

  //loops while the program runs. update and publish both necessary for all nodes.
  while (ros::ok()) {
    test->update();
    test->publish();
  }
}
