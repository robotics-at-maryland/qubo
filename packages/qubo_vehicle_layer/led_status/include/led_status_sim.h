#ifndef LED_HEADER
#define LED_HEADER

#include "qubo_node.h"

class LedSimNode : QuboNode {
 protected:
  std::string name;
  bool on;

 public:
  LedSimNode(int,char**,int);
  ~LedSimNode();

  void update();
  void publish();

};

#endif
