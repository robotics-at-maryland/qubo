#include "qscu_node.h"


int main(int argc, char* argv[]){
	ros::init(argc, argv, "qscu_control_node");
	ros::NodeHandle nh;
	QSCUNode qscu_node(nh, "qscu_node", "/dev/ttyACM0");

	qscu_node.update();

	return 0;
}
