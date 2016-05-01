#ifndef LED_HEADER
#define LED_HEADER

#include <string>
#include "qubo_node.h" //always included
#include "ram_msgs/Led.h" //specialized ram_msg included

#define DEFAULT_ENABLED false

class LedSimNode : public QuboNode {
 protected: //fields and private methods
  std::string ledName;
  ram_msgs::Led msg; //creates an instance of the message to put info in
  bool enabled; //any 

 public: //public methods
  LedSimNode(int,char**,int,std::string); //first 3 fields are mandatory; 4th for any node-specific fields
  ~LedSimNode(); //destructor; ignore

  //update: provides any info from other nodes that is needed for this node's operation
  void update();
  //publish: puts information from this node into the ram_msg
  void publish();
  //enable: turns the LED on
  void enable();
  //disable: turns the LED off
  void disable();

};

#endif
