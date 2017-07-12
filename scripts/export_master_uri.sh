#sgillen@20175203-09:52 I made this script mostly to remind myself of the syntax for exporting the ROS_MASTER_URI variable
#it's janky, but just uncomment the line for the URI you want. 


#uncomment this line to set the master URI to your localhost, this should work for everyone
export ROS_MASTER_URI=http://localhost:11311

#this one is for me (sean) should work for other people using
#export ROS_MASTER_URI=http://192.168.129.1:11311


#in general yours should look like
#export ROS_MASTER_URI=http://<ip_of_other_machine>:11311
