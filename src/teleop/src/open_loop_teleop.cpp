#include "open_loop_teleop.h"

/*
 * Default OpenLoopTeleop constructor. Initializes a subscriber object to subscribe to raw input from the joystick
 * and call the joyInputCallback callback function. Initializes a publisher object to publish mapped data to the
 * thrusters. Sets up the thruster input message with the proper dimensions and sets data array to all 0.
 */
OpenLoopTeleop::OpenLoopTeleop(int rate) {
    // Subscribe to raw input from the joystick and publish to thruster input
    joy_sub = nh.subscribe<sensor_msgs::Joy>("/joy_input", rate, &OpenLoopTeleop::joyInputCallback, this);
    thruster_pub = nh.advertise<std_msgs::Float64MultiArray>("/tortuga/thruster_input", 10);
    
    // Initialize the thruster input message
    thruster_input.layout.dim.resize(1);
    thruster_input.data.resize(6);
    thruster_input.layout.dim[0].label = "thruster_input";
    thruster_input.layout.dim[0].size = 6;
    thruster_input.layout.dim[0].stride = 4;
    for (int i = 0; i < thruster_input.layout.dim[0].size; i++) {
        thruster_input.data[i] = 0;
    }
}

/*
 * Default OpenLoopTeleop destructor.
 */
OpenLoopTeleop::~OpenLoopTeleop(){}

/*
 * Updates the thrusters by publishing the thruster input message to the thruster input topic.
 */
void OpenLoopTeleop::update() {
    thruster_pub.publish(thruster_input);
}

/*
 * Parses the data from the joystick's raw input and stores it in the thruster input message. Currently only
 * allows for driving the robot forward/backward and turning it left/right.
 *
 * TODO: Add mappings so the robot can dive.
 */
void OpenLoopTeleop::joyInputCallback(const sensor_msgs::Joy::ConstPtr &joy_input) {
    // Set the two forward thrusters for now
    // Note: Need to set correct addition/subtraction for joy_input->axes[0]
    thruster_input.data[0] = (-joy_input->axes[1] - joy_input->axes[0]) / 2 * MAX_THRUSTER_INPUT;
    thruster_input.data[1] = (-joy_input->axes[1] + joy_input->axes[0]) / 2 * MAX_THRUSTER_INPUT;
}

/*
 * Initializes an open loop teleoperation node and publishes thruster input messages at a rate of 10 Hz
 * while the node is still active.
 */
int main(int argc, char **argv) {
    // Initialize the teleop node and node handle
    ros::init(argc, argv, "teleop_node");
    OpenLoopTeleop node(10);

    // Perform callbacks and publish thruster input while node is active
    ros::Rate rate(10);
    while (ros::ok()) {
        ros::spinOnce();
        node.update();
        rate.sleep();
    }

    return 0;
}
