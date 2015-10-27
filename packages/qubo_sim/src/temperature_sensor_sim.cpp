#include "ros/ros.h"
#include "ram_msgs/Sim_Temperature.h"

int main(int argc, char **argv) {
  ros::init(argc, argv, "simulated_temperature_sensor");
  
  ros::NodeHandle n;
  ros::Publisher temperature_sensor = n.advertise<ram_msgs::Sim_Temperature>("/qubo/temperature", 100);

  ros::Rate loop_rate(100);

  while(ros::ok()) {
      ram_msgs::Sim_Temperature temperature_msg;
      temperature_msg.heat = 23.0;

      temperature_sensor.publish(temperature_msg);

      ros::spinOnce();
      loop_rate.sleep();
  }
}
