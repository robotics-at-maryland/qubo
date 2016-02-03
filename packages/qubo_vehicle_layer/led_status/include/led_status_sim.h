#ifndef LED_HEADER
#define LED_HEADER

#include <string>
#include "qubo_node.h"
#include "ram_msgs/Led.h"

#define DEFAULT_ENABLED false

class LedSimNode : QuboNode {
 protected:
  std::string ledName;
  ram_msgs::Led msg;
  bool enabled;

 public:
  LedSimNode(int,char**,int,std::string);
  ~LedSimNode();

  void update();
  void publish();
  void enable();
  void disable();

};

#endif
