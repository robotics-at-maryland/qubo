#include "ros/ros.h"

#include <iostream>
#include <string>

/*
 * This file sets parameters in the parameter server that represents
 * "house keeping" values for the hull passed as a parameter. All values that this
 * program sets should be relatively constant.
 *
 * Usage: simulated_hull_sensor HULL PRESSURE TEMPERATURE HUMIDITY
 * Parameter names:
 *      pressure: /qubo/HULL/pressure
 *      temperature: /qubo/HULL/temperature
 *      humidity: /qubo/HULL/humidity
 */

/*
 * A short utility method to set a ros parameter with a float value and record
 * relevant log messages. 
 */
void setParameter(std::string parameterPath, float value){
    /*
     * If the parameter was previously set, log a message detailing this.
     */
    float prev;
    if(ros::param::get(parameterPath,prev)){
        ROS_INFO("parameter %s is already set to %f. This will be overridden.",parameterPath.c_str(),prev);
    }

    /*
     * Use the method provided by ros to actually set the parameter on the
     * parameter server.
     */
    ros::param::set(parameterPath,value);
    ROS_INFO("parameter %s set to %f",parameterPath.c_str(),value);
}

int main(int argc, char **argv) {
    /*
     * If less than the required number of args is passed we print a helpful
     * message detailing the command line args.
     */
    if(argc < 4){
        std::cout << "Usage: simulated_hull_sensor HULL PRESSURE TEMPERATURE HUMIDITY\n";
        std::cout << "Parameter names:\n";
        std::cout << "\tpressure: /qubo/HULL/pressure\n";
        std::cout << "\ttemperature: /qubo/HULL/temperature\n";
        std::cout << "\thumidity: /qubo/HULL/humidity\n";
        return 1;
    }

    /*
     * initializes this file with ROS. Even though this is not a publisher or
     * subscriber this is requited to let this file to be run through rosrun.
     */
    ros::init(argc, argv, "simulated_hull_sensor");


    /*
     * Initialize variables with the values contained in the command line args
     */
    std::string hull = argv[1];
    float pressure = std::atof(argv[2]);
    float temperature = std::atof(argv[3]);
    float humidity = std::atof(argv[4]);

    /*
     * use these values to set parameters on the ROS parameter server.
     */
    ROS_INFO("Setting parameters for hull %s",hull.c_str());
    setParameter("/qubo/"+hull+"/pressure",pressure);
    setParameter("/qubo/"+hull+"/temperature",temperature);
    setParameter("/qubo/"+hull+"/humidity",humidity);

    return 0;
}