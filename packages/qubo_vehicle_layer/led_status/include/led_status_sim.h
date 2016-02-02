#ifndef LED_HEADER
#define LED_HEADER

#include <string>
#include "qubo_node.h"
#include "ram_msgs/Led.h"

class LedSimNode : QuboNode {
 protected:
  std::string ledName;
  bool enabled;

 public:
  LedSimNode(int,char**,int,std::string);
  ~LedSimNode();

  void update();
  void publish();

};

#endif
