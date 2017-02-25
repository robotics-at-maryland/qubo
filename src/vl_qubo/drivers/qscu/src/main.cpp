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
    
    
}
