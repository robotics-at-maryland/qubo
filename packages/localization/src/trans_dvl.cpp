#include <ros/ros.h>
#include <geometry_msgs/TwistWithCovarianceStamped.h>
#include <underwater_sensor_msgs/DVL.h>

ros::Publisher pub;
geometry_msgs::TwistWithCovarianceStamped msg;

void convertMessage(const underwater_sensor_msgs::DVL underwater_msg) {
  msg.header.frame_id = "base_link";
  msg.header.stamp = ros::Time::now();
        
  msg.twist.twist.linear.x = underwater_msg.bi_x_axis;
  msg.twist.twist.linear.y = underwater_msg.bi_y_axis;
  msg.twist.twist.linear.z = underwater_msg.bi_z_axis;
        
  msg.twist.covariance = {1e-9, 0, 0, 0, 0, 0,
                          0, 1e-9, 0, 0, 0, 0,
                          0, 0, 1e-9, 0, 0, 0,
                          0, 0, 0, 1e-9, 0, 0,
                          0, 0, 0, 0, 1e-9, 0,
                          0, 0, 0, 0, 0, 1e-9};
        
  pub.publish(msg);
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "trans_dvl");
  ros::NodeHandle n;
  
  pub = n.advertise<geometry_msgs::TwistWithCovarianceStamped>("qubo/twist", 1000);
  
  ros::Subscriber sub = n.subscribe("qubo/dvl", 1000, convertMessage);
  ros::spin();
  
  return 0;
}
