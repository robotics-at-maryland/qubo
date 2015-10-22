#include "ros/ros.h"
#include "ram_msgs/Sim_Battery.h"

#include <sstream>

#define DEFAULT_VOLTAGE 24
#define DEFAULT_CURRENT 9001

int main(int argc, char **argv) {
    /* 
     * ros::init(argc, argv, "<NAME OF NODE>");
     */
    
    double curr_voltage = DEFAULT_VOLTAGE;
    double curr_current = DEFAULT_CURRENT;

    ros::init(argc, argv, "simulated_battery_sensor");
    ros::NodeHandle n;

    ros::Publisher battery_sensor = n.advertise<ram_msgs::Sim_Battery>("/qubo/battery", 100);

    ros::Rate loop_rate(100);

    while(ros::ok()) {
        ram_msgs::Sim_Battery battery_msg;
        battery_msg.voltage = curr_voltage;
        battery_msg.current = curr_current; 
        battery_msg.batt1 = 0;
        battery_msg.batt2 = 0;
        battery_msg.shore = 0;

        battery_sensor.publish(battery_msg);
        ros::spinOnce();
        loop_rate.sleep();
    }
}
