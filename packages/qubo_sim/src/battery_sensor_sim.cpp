#include "ros/ros.h"
#include "ram_msgs/Sim_Power_Source.h"

#include <sstream>

#define DEFAULT_SOURCE 0
#define DEFAULT_VOLTAGE 24.0
#define DEFAULT_CURRENT 9001.0

int main(int argc, char **argv) {
    /* 
     * ros::init(argc, argv, "<NAME OF NODE>");
     * 
     * By running this file's executable, ros is starting up the executable as a node.
     * init() function initializes the executable as a node. Make sure to use unique names
     * for every node that you intend on starting.
     */
    ros::init(argc, argv, "simulated_battery_sensor");

    /*
     * NodeHandle is the main access point to communicate with the ros system.
     */
    ros::NodeHandle n;
    
    /*
     * This creates a Publisher object called battery_sensor. This advertise() function is how you tell
     * ros what you want to publish to a specific topic name.
     * This function will also return a Publisher object which allows you to post messages to the
     * specified topic.
     * Second paramter of the advertise() function is the size of the message queue that will be used
     * for publishing messages. If messages are published faster than we can send them, the number here
     * Specified how many messages to buffer.
     */
    ros::Publisher battery_sensor = n.advertise<ram_msgs::Sim_Power_Source>("/qubo/battery", 100);

    /*
     * This specifies what frequency that you want the loop to run at. This works in conjunction with
     * loop_rate.sleep() at the end of the while loop.
     */
    ros::Rate loop_rate(100);
    
    /*
     * Initializes a source_source param for the parameter server to allow
     * live modifications to default values.
     */
    n.setParam("/qubo/power_source", DEFAULT_SOURCE);
    n.setParam("/qubo/power_enabled", true);
    n.setParam("/qubo/voltage", DEFAULT_VOLTAGE);
    n.setParam("/qubo/current", DEFAULT_CURRENT);

    /*
     * While ros is A OK this loop will keep rollin'
     */
    while(ros::ok()) {
        /*
         * Creates the message objects for you to stuff full of data.
         */
        ram_msgs::Sim_Power_Source msg;
        
        /*
         * Temp variables to hold onto current parameter data.
         */
        int source;
        bool enabled;
        double curr_voltage;
        double curr_current;
        
        /*
         * Checks parameter server for new paramters to set.
         */
        n.param("/qubo/power_source",source, DEFAULT_SOURCE);
        n.param("/qubo/power_enabled",enabled, true);
        n.param("/qubo/voltage",curr_voltage, DEFAULT_VOLTAGE);
        n.param("/qubo/current",curr_current, DEFAULT_CURRENT);
        
        /*
         * Sets the message contents to the variables above.
         */
        msg.source = source;
        msg.enabled = enabled;
        msg.voltage = curr_voltage;
        msg.current = curr_current;
        
        /*
         * This actually posts the message object to the topic. Note how it's
         * using the Publisher object initialized before hand.
         */
        battery_sensor.publish(msg);

        /*
         * Not sure exactly what this is used for. Will edit this part when 
         * We need to use it. Apparently its used if this file doubles as a
         * subscriber to catch callbacks?
         */
        ros::spinOnce();

        /*
         * Used in conjunction with loop_rate.
         */
        loop_rate.sleep();
    }
}
