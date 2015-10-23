class QuobNode {
  bool simulated;
  
 Public:
  virtual void subscribe() = 0;
  virtual void publish() = 0;
  virtual void sendAction();
 
}
