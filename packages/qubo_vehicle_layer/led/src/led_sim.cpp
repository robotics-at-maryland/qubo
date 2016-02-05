#include "led_sim.h"

LedSimNode::LedSimNode(int argc, char **argv, int rate, std::string name) {
  ledName = name;
  enabled = false;
  publisher = n.advertise<ram_msgs::Led>("/qubo/led/" + ledName, 1000);
  n.setParam("qubo/led/" + ledName + "/enabled", DEFAULT_ENABLED);
}

LedSimNode::~LedSimNode() {}

void LedSimNode::update() {
  ros::spinOnce();
}

void LedSimNode::publish() {
  msg.led_name = ledName;
  msg.enabled = enabled;
  publisher.publish(msg);
}

void LedSimNode::enable() {
  enabled = true;
}

void LedSimNode::disable() {
  enabled = false;
}
