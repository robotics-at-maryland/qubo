#include "qscu_control_node.h"


int main(int argc, char* argv[]){
	ros::init(argc, argv, "qscu_control_node");
	ros::NodeHandle nh;
	QSCUControlNode qscu_node(nh, "qscu_node", "/dev/ttyACM0");

	qscu_node.update();

	return 0;
}
