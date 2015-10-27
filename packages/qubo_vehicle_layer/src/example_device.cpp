#include "device.h"

class ExampleDevice : public Device {

	ros::Publisher publisher = 
		n.advertise<ram_msg::msg::Vector2>("example", 1000);
	ros::Rate loop_rate(10);

public:
	
	void publish(){
		ram_msg::msg::Vector2 v;
		v.x = 5.0;
		publisher.publish(v);
		ros::spinOnce();
		loop_rate.sleep();
	}

	void subscribe(){

	}

	int main(int argc, int **argv){
		ros::init(argc, argv, "ExampleDevice");
		ros::NodeHandle n;

		while(ros::ok){
			publish();
		}
		return 0;
	}

};