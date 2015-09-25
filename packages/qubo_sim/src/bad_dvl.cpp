#include "bad_dvl.h"

//please don't emulate this

using namespace std;
int main(){
  ofstream fout;
  fout.open ("example.dat");
    
  for(int i = 0; i < 1000; i++){
    fout << rand() % 100; //should give a random number between 0-100
  }


 fout.close();
 return 0;
  
}
