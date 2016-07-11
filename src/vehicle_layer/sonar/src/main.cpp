

int main(int argc, char **argv){

  if(argc != 2){
    ROS_ERROR("Incorrect number of arguments")
    exit(1);
  }

  std::shared_ptr<ros::NodeHandle> n1;
  std::shared_ptr<ros::NodeHandle> n2;
  ros::init(argc, argv, "sonar_node");

  std::unique_ptr<RamNode> client_node;
  std::unique_ptr<RamNode> server_node;

  if(!strcmp(argv[1],"data")){
    client_node.reset(new sonar_client(n1, 10));
    server_node.reset(new sonar_server(n2, 10));
  }


  while(ros::ok()){
    node->update();
  }
}
