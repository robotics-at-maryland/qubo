nclude "ros/ros.h"
  28 #include "std_msgs/String.h"
  29 
  30 #include <sstream>
  31 
  32 /**
  33  * This tutorial demonstrates simple sending of messages over the ROS system.
  34  */
  35 int main(int argc, char **argv)
  36 {
  37   /**
  38    * The ros::init() function needs to see argc and argv so that it can perform
  39    * any ROS arguments and name remapping that were provided at the command line.
  40    * For programmatic remappings you can use a different version of init() which takes
  41    * remappings directly, but for most command-line programs, passing argc and argv is
  42    * the easiest way to do it.  The third argument to init() is the name of the node.
  43    *
  44    * You must call one of the versions of ros::init() before using any other
  45    * part of the ROS system.
  46    */
  47   ros::init(argc, argv, "talker");
  48 
  49   /**
  50    * NodeHandle is the main access point to communications with the ROS system.
  51    * The first NodeHandle constructed will fully initialize this node, and the last
  52    * NodeHandle destructed will close down the node.
  53    */
  54   ros::NodeHandle n;
  55 
  56   /**
  57    * The advertise() function is how you tell ROS that you want to
  58    * publish on a given topic name. This invokes a call to the ROS
  59    * master node, which keeps a registry of who is publishing and who
  60    * is subscribing. After this advertise() call is made, the master
  61    * node will notify anyone who is trying to subscribe to this topic name,
  62    * and they will in turn negotiate a peer-to-peer connection with this
  63    * node.  advertise() returns a Publisher object which allows you to
  64    * publish messages on that topic through a call to publish().  Once
  65    * all copies of the returned Publisher object are destroyed, the topic
  66    * will be automatically unadvertised.
  67    *
  68    * The second parameter to advertise() is the size of the message queue
  69    * used for publishing messages.  If messages are published more quickly
  70    * than we can send them, the number here specifies how many messages to
  71    * buffer up before throwing some away.
  72    */
  73   ros::Publisher chatter_pub = n.advertise<std_msgs::String>("chatter", 1000);
  74 
  75   ros::Rate loop_rate(10);
  76 
  77   /**
  78    * A count of how many messages we have sent. This is used to create
  79    * a unique string for each message.
  80    */
  81   int count = 0;
  82   while (ros::ok())
  83   {
  84     /**
  85      * This is a message object. You stuff it with data, and then publish it.
  86      */
  87     std_msgs::String msg;
  88 
  89     std::stringstream ss;
  90     ss << "hello world " << count;
  91     msg.data = ss.str();
  92 
  93     ROS_INFO("%s", msg.data.c_str());
  94 
  95     /**
  96      * The publish() function is how you send messages. The parameter
  97      * is the message object. The type of this object must agree with the type
  98      * given as a template parameter to the advertise<>() call, as was done
  99      * in the constructor above.
 100      */
 101     chatter_pub.publish(msg);
 102 
 103     ros::spinOnce();
 104 
 105     loop_rate.sleep();
 106     ++count;
 107   }
 108 
 109 
 110   return 0;
 111 }
