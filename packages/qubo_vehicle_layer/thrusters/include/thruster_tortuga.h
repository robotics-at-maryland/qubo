#ifndef THRUSTER_TORTUGA_H

#define THRUSTER_TORTUGA_H


#define NUM_THRUSTERS 6

#include "tortuga_node.h"
#include "sensor_msgs/Joy.h"
#include "std_msgs/Int64MultiArray.h"
#include "tortuga/sensorapi.h"


class ThrusterTortugaNode : public TortugaNode {
  
 public:
  ThrusterTortugaNode(int, char**, int);
  ~ThrusterTortugaNode();
  
  void update();
  void publish();
  void thrusterCallBack(const std_msgs::Int64MultiArray msg);
  
    protected:
    
    /* JW:
     * Here is the definition for the Tortuga specific Float64MultiArray
     * used to control the thruster power.
     *
     * See: http://docs.ros.org/jade/api/std_msgs/html/msg/Float64MultiArray.html
     * for more information
     * 
     * Every Float64MultiArray includes a "layout" member that specifies
     * the format of the data in the array.  Here's how our "layout" 
     * should be set up:
     * 
     * dim[0].label = "thruster" 	| descriptor of the data
     * dim[0].size = 6 				| amount of elements in the array
     * dim[0].stride = 6				| amount of elements between succsesive 
     *								|	indexes.  See the website above for
     *								|	more information
     * data_offset = 0				| no offset for now
     *
     *
     * multiarray(i,j,k) = data[data_offset + dim_stride[1]*i + dim_stride[2]*j + k]
     * in our case:
     *	 multiarry(i) = data[data_offset + i]
     *
     * To set thruster 2 to "0.50":
     * 	data[1] = 0.50
     */
    
    

  //contains the current powers we want the thrusters to be operating at
  //this is the DESIRED relative power, since our thrusters our nonlinear
  //we'll need to map these to another vector eventually. 
  std_msgs::Int64MultiArray thruster_powers;
  int fd;
  std::string sensor_file;
  ros::Subscriber subscriber;

};

#endif
