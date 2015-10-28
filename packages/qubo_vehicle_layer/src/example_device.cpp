#include "vehicle_layer.h"
#include "QuboNode.cpp"
#include "ram_msgs/Vector2.h"

class ExampleDevice : public QuboNode {

protected:
	

public:
	
	ExampleDevice(){
		ExampleDevice::publisher = pub_node.advertise<ram_msgs::Vector2>("example", 1000);
		ExampleDevice::rate = ros::Rate(10);
	}

	~ExampleDevice(){
		//nothing here
	}

	void publish(){
		ram_msgs::Vector2 v;
		v.x = 5.0;
		publisher.publish(v);
		ros::spinOnce();
		rate.sleep();
	}

	void subscribe(){
		//nothing
	}

	void sendAction(){
		//nothing
	}

};

int main(int argc, char **argv){
		ExampleDevice* ED = new ExampleDevice();
		ros::init(argc, argv, "ExampleDevice");

		while(ros::ok){
			(*ED).publish();
		}

		return 0;
	}