#include "device.h"


//default abstract class that all devices in the vehicle_layer
//should inherit from
//not actually written yet
virtual class Device{
public:
		/*
		data
		publish
		subscribe
		push
		*/

		void publish();
		void subscribe();

	int main(int arga, char **argv){

		ros::init(argc, argv, "virtual_device");
		ros::NodeHandle n;
	}
private:

}