
#include "../include/util.h"
#include <string>
#include <iostream>
#include <unistd.h>
#include <time.h>

int main(int argc, char *argv[])
{
    if (argc == 3)
    {
        DVL *dvl = NULL;
        DVL::DVLData data;
        DVL::SystemConfig config;
        DVL::DataConfig output = {};
        clock_t curr, last;
        double hz;
        config.speed = DVL::k115200;
        config.parity = DVL::Parity::NO_PARITY;
        config.two_stopbits=false;
        config.auto_ensemble_cycling=true;
        config.auto_ping_cycling=true;
        config.binary_data_output=true;
        config.serial_output=true;
        config.turnkey=true;
        config.recorder_enable=false;
        output.output_type = DVL::DataOutput::ALL_DATA;
        //output.output_type = DVL::DataOutput::PARTIAL_DATA;
        //output.output_type = DVL::DataOutput::MINIMUM_DATA;
        //output.output_type = DVL::DataOutput::TEXT_DATA;
        output.profile_output[DVL::VELOCITY] = true,
        output.profile_output[DVL::CORRELATION] = true,
        output.profile_output[DVL::ECHO_INTENSITY] = true,
        output.profile_output[DVL::PERCENT_GOOD] = true,
        output.profile_output[DVL::STATUS] = true;
        try {
            dvl = new DVL(std::string(argv[1]),getBaudrate(argv[2]));
            dvl->openDevice();
            dvl->loadUserSettings();
            dvl->setSystemConfiguration(config);
            dvl->setDataTransferConfiguration(output);
            dvl->enableMeasurement();
            last = clock();
            while (dvl->isOpen()) {
                curr = clock();
                data = dvl->getDVLData();
                std::cout 
                    << data.mms_east << "/" 
                    << data.mms_north << "/" 
                    << data.mms_surface << std::endl;
                hz = CLOCKS_PER_SEC / ((double)(curr-last))/10;
                std::cout << "Polling at " << hz << " Hz" << std::endl;
                last = curr;
            }
        } catch (std::exception& e) {
            dvl->closeDevice();
            delete dvl;
            printError(e);
            return DVL_ERR;
        }
        if (dvl != NULL) {
            dvl->closeDevice();
            delete dvl;
        }
        return 0;
    }
    printUsage();
}
