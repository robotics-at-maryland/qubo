#include "ros/ros.h"
#include "ram_msgs/Sim_Temperature.h"

#include <random>

#define DEFAULT_MEAN_TEMP 23.0

int main(int argc, char **argv) {
    ros::init(argc, argv, "simulated_temperature_sensor");
    
    ros::NodeHandle n;
    ros::Publisher temperature_sensor = n.advertise<ram_msgs::Sim_Temperature>("/qubo/temperature", 100);

    ros::Rate loop_rate(100);


    /*
     * Initialize a ROS parameter server param to the default temperature (approx. room temp)
     */
    n.setParam("/simulated_temperature_sensor/mean_temperature",DEFAULT_MEAN_TEMP);

    /*
     * Set up a RNG that will be used to add varience to the published temperatures.
     */
    double lower_bound = -1;
    double upper_bound = 1;
    std::uniform_real_distribution<double> unif(lower_bound,upper_bound);
    std::default_random_engine re;

    while(ros::ok()) {
        ram_msgs::Sim_Temperature temperature_msg;

        /*
         * Retreive the stored mean_temperature from the parameter server. If the parameter
         * cannot be found, DEFAULT_MEAN_TEMP is used instead.
         */
        double mean_temperature;
        n.param("/simulated_temperature_sensor/mean_temperature",mean_temperature,DEFAULT_MEAN_TEMP);

        /*
         * The temperature published in the message is the mean_temperature adjusted with
         * some noise.
         */
        temperature_msg.heat = mean_temperature + unif(re);

        temperature_sensor.publish(temperature_msg);

        ros::spinOnce();

        loop_rate.sleep();
    }
}
