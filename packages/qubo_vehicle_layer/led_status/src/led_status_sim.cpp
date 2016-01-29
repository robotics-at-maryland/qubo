#include "led_status_sim.h"

LedSimNode::LedSimNode(int argc, char **argv, int rate, std::string name) {
  ledName = name;
  enabled = false;
}

LedSimNode::~LedSimNode() {}

void LedSimNode::update() {
  ros::spinOnce();
}

void LedSimNode::publish() {

}
