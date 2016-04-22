/*This class takes joystick inputs and converts them to velocity in R3. */

ThrusterTortugaNode::ThrusterTortugaNode(int argc, char **argv, int rate): TortugaNode(){
    ros::Rate loop_rate(rate);
    subscriber = n.subscribe("/joy", 1000, &ThrusterTortugaNode::thrusterCallBack, this);
}

ThrusterTortugaNode::~ThrusterTortugaNode(){
    setSpeeds(fd, 0, 0, 0, 0, 0, 0);
    //SG: does close make sense there?
    close(fd);
}

void ThrusterTortugaNode::update(){
    //I think we need to initialize thrusters and stuff before this will work 
    ros::spinOnce();
    setSpeeds(fd, msg.data[0], msg.data[1], msg.data[2], msg.data[3], msg.data[4], msg.data[5]);
}

void ThrusterTortugaNode::publish(){
    //   setSpeeds(fd, msg.data[0], msg.data[1], msg.data[2], msg.data[3], msg.data[4], msg.data[5]);
}

void ThrusterTortugaNode::thrusterCallBack(const std_msgs::Float64MultiArray new_vector){
    //SG: TODO change this shit
    msg.data = new_vector.data;
}



