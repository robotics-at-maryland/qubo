#include "vehicle_layer.h"
#include "vehicle_node.h"
#include "ram_msgs/Vector2.h"

class ExampleDevice : public QuboNode {

protected:
	

public:
	
	ExampleDevice(){
		ExampleDevice::publisher = pub_node.advertise<ram_msgs::Vector2>("example", 1000);
		//ExampleDevice::rate = ros::Rate(10);
	}

	~ExampleDevice(){
		printf("hello world");
		//nothing here
	}

	void publish(){
		ram_msgs::Vector2 v;
		v.x = 5.0;
		publisher.publish(v);
		ros::spinOnce();
		//rate.sleep();
	}

	void subscribe(){
		//nothing
	}

	void sendAction(){
		//nothing
	}

};

