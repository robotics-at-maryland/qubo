#include "ros/ros.h"
#include "ram_msgs/Sim_Led.h"

// For now only LED is the battery level as an int representing a percentage
#define DEFAULT_BATTERY_LEVEL 100

int main(int argc, char **argv) {

	ros::init(argc, argv, "simluated_led_sensor");

	// Initializing the publisher
	ros::NodeHandle n;
	ros::Publisher led_sensor = n.advertise<ram_msgs::Sim_Led>("qubo/led_status",100);

	ros::Rate loop_rate(100);

	// Setting the parameter server's battery_level paramter the default value
	n.setParam("/simulated_led_sensor/battery_level", DEFAULT_BATTERY_LEVEL);

	while(ros::ok()) {
		ram_msgs::Sim_Led led_msg;

		int bat_lvl;
		// Get the battery_level parameter from the parameter server
		n.param("/simulated_led_sensor/battery_level", bat_lvl, DEFAULT_BATTERY_LEVEL);

		// Sets the message to be published as the value from the parameter server
		led_msg.battery_level = bat_lvl;
		led_sensor.publish(led_msg);

		ros::spinOnce();

		loop_rate.sleep();
	}
}
