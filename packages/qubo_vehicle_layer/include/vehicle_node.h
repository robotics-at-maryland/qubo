class QuboNode {
 public:
  QuboNode(); //Constructor, you should really never call this directly
  ~QuboNode(); //Destructor 

  virtual void subscribe() = 0;
  virtual void publish() = 0;
  virtual void sendAction();  //this isn't a pure function because sub classes won't necessarily use it. 


  //We'll probably need a few more things 
 protected:
  ros::NodeHandle pub_node; //the handle for the publisher node
  ros::Publisher publisher;
  ros::Rate rate;
  
};
