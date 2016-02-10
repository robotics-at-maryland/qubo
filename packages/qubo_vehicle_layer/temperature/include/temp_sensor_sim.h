#ifndef TEMPSIM_HEADER
#define TEMPSIM_HEADER

#include <string>
#include <random>
#include "qubo_node.h" //always included
#include "ram_msgs/Temperature.h" //include ram msg file specific to node

#define DEFAULT_TEMP 23.0

class TempSimNode : QuboNode {
 protected: //fields
  std::string sensorName;
  ram_msgs::Temperature msg; //always include this, used to create specific message file for this node
  double real_temp;
  double lower_bound = -1;
  double upper_bound = 1;
  // std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
  // std::default_random_engine re;

 public: //public methods
  TempSimNode(int, char **, int, std::string); //constructor: first three fields mandatory, then specific fields to the node
  ~TempSimNode(); //destructor, necessary
  
  //update: retrieves data from any other node needed for operation.
  void update();
  //publish: puts data about the node in the message file.
  void publish();
};

#endif
