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
    qscu.openDevice();
    if(qscu.isOpen()){
        std::cout << "well something appeared to work" << std::endl;
    }
    std::cout << std::flush;
    struct Depth_Status d_s;
    Transaction t_s = tDepthStatus;
    std::cout << "writing message" <<std::endl;
    qscu.sendMessage(&t_s, NULL, &d_s);
    printf("received: %f, %i", d_s.depth_m, d_s.warning_level);

}
