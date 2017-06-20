// UNIX Serial includes
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <stdarg.h>

// Header include
#include "QSCU.h"


#include <stdio.h>
#include <iostream>

int main(){
  QSCU qscu("/dev/ttyACM0", B115200);
  reconnect:
  do {
    try {
      qscu.openDevice();
	} catch (const QSCUException e) {
      std::cout << e.what() << std::endl;
    }
  } while (!qscu.isOpen());

  if(qscu.isOpen()){
    std::cout << "well something appeared to work" << std::endl;
  }
  std::cout << std::flush;
  Message alive;
  while ( true ){
    std::cout << "emitting keepAlive" << std::endl;
	if ( qscu.keepAlive() ){
	  std::cout << "keepAlive failed" << std::endl;
	} else {
	  std::cout << "keepAlive success" << std::endl;
	  // sleep for half a second, then repeat
	  usleep(250000);
	}
	std::cout << "checking status" << std::endl;
	Transaction t_e = tEmbeddedStatus;
	struct Embedded_Status e_s;
	qscu.sendMessage(&t_e, NULL, &e_s);
	std::cout << "Response: mem - " << e_s.mem_capacity << " uptime - " << e_s.uptime << std::endl;
  }
  struct Depth_Status d_s;
    Transaction t_s = tDepthStatus;
    std::cout << "writing message" <<std::endl;
    qscu.sendMessage(&t_s, NULL, &d_s);
    printf("received: %f, %i", d_s.depth_m, d_s.warning_level);

}
