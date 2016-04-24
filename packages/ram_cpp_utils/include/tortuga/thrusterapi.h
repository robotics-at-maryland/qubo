
/* In msec */
#define IO_TIMEOUT  100

// If we are compiling as C++ code we need to use extern "C" linkage
#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/** Returns the file*/
int openThrusters(const char * devName);
int setSpeed(int fd, int addr, int speed);
int multiCmd(int fd, int cmd, int addr, unsigned char * data, int len,
             int timeout);
int setReg(int fd, int addr, int reg, int val);

#define TH_IOERROR -1
#define TH_TIMEOUTERROR -1


#define REG_ADDR  7
#define REG_TIMER 9

// If we are compiling as C++ code we need to use extern "C" linkage
#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus
