#include "imu_sim.h"


int main(int argc, char **argv){

  ros::init(argc, argv, "imu_node"); //basically always needs to be called first
  bool simulated = true; //We'll have to pass this one in eventually 


  //if(simulated){
  ImuSimNode *node = new ImuSimNode(argc, argv, 10); //10 (the rate) is completely arbitary
  // }
  
  while (ros::ok()){
    node->update();
    node->publish();
  }

}
