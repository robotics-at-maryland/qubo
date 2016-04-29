// STD Includes
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

// Project Includes
#include "tortuga/sensorapi.h"

#ifndef SENSORAPI_R5
#error "WRONG VERSION OF INCLUDE"
#endif

/*
 * Sensor Board Utility, Revision $rev:$
 *
 */

void thrusterCmd(int fd, int cmd)
{
    if(setThrusterSafety(fd, cmd) != SB_OK)
        printf("Error safing thruster\n");
}

void barCmd(int fd, int cmd)
{
    if(setBarState(fd, cmd) != SB_OK)
        printf("Error setting bar state\n");
}


int main(int argc, char ** argv)
{
    if(argc < 2 || ((argc == 2) && (strcmp(argv[1], "-h") == 0) ))
    {
        printf("LCD-related Commands:\n");
        printf("\t-c (clear screen)\n");
        printf("\t-t line1 line2\n");
        printf("\t-bloff (backlight on)\n");
        printf("\t-blon (backlight off)\n");
        printf("\t-blfl (backlight flash)\n");

        printf("\nSafety commands:\n");
        printf("\t-safe (disable thrusters)\n");
        printf("\t-unsafe (enable thrusters)\n");
        printf("\t-safe [n] (safe thruster n)\n");
        printf("\t-unsafe [n] (unsafe thruster n)\n");
        printf("\t-diagon  (runtime diagnostics on)\n");
        printf("\t-diagoff (runtime diagnostics off)\n");
        printf("\t-tstop (send zero speed command)\n");

        printf("\nPower commands:\n");
        printf("\t-hardkill  (kill power)\n");
        printf("\t-extpower  (switch to external power source)\n");
        printf("\t-intpower  (switch to batteries)\n");
        printf("\t-batton n (enable battery n)\n");
        printf("\t-battoff n  (disable battery n)\n");
        printf("\t-setovr a b  (configure overcurrent parameters)\n");
        printf("\t-readovr  (read overcurrent parameters)\n");
        printf("\t-setspeed a b c d e f  (set speeds)\n");

        printf("\nSonar commands:\n");
        printf("\t-bfstart (starts the Blackfin processor)\n");
        printf("\t-bfstop (stops the Blackfin processor)\n");
        printf("\t-bfreset (reboots the Blackfin processor)\n");
        printf("\t-sonar (retrieve latest sonar telemetry)\n");
        
        printf("\nOther commands:\n");
        printf("\t-check (crude system check)\n");
        printf("\t-status (show sensor readings)\n");
        printf("\t-marker {1|2} (drop marker 1 or 2)\n");
        printf("\t-s  (begin start sequence)\n");

        printf("\nLED Bar commands (DEPRICATED)");
        printf("\t-baron (enable bar outputs)\n");
        printf("\t-baroff (disable bar outputs)\n");
        printf("\t-baron [n] (enable bar output n)\n");
        printf("\t-baroff [n] (disable bar output n)\n");
     
        printf("\t-setbars n (set bar outputs)\n");
        printf("\t-noblink (stop animation)\n");
        printf("\t-redgreen (start red/green animation)\n");
        printf("\t-redblue (start red/blue animation)\n");

        printf("\nDVL commands:\n");
        printf("\t-dvlon (enable dvl)\n");
        printf("\t-dvloff (disable dvl)\n");

        printf("\nPneumatics Commands:\n");
        printf("\t-firetorp # (fire torpedo number #)\n");
        printf("\t-armtorp # (arm torpedo number #)\n");
        printf("\t-voidtorp # (void torpedo number #)\n");
        printf("\t-extgrabber (close the grabber)\n");
        printf("\t-retgrabber (open the grabber)\n");
        printf("\t-voidgrabber (void the grabber)\n");
        printf("\t-voidall (void all pneumatics)\n");
        printf("\t-pneuoff (turn off pneumatics)\n");

        printf("\nDerpy Commands:\n");
        printf("\t-derpypwr {on|off} (turn on or off derpy power)\n");
        printf("\t-setderpy # (set derpy speed)\n");
        printf("\t-stopderpy (set derpy speed to not move)\n");

        printf("\nCamera Relay Board Commands:\n");
        printf("\t-camconnect (connect cameras)\n");
        printf("\t-camdisconnect (disconnect cameras)\n");


        return -1;
    }

    int fd = openSensorBoard("/dev/sensor");

    if(fd == -1)
    {
        printf("\nCould not find device!\n");
        printf("Is the right device set?\n");
        printf("Has the serial number of this FTDI chip been added to udev?\n");
        printf("Because that needs to be done for every board we make.\n");

        close(fd);
        return -1;
    }

    if(syncBoard(fd) != 0)
    {
        printf("\nCould not sync with board!\n");
        close(fd);
    }

	if(strcmp(argv[1], "-readovr") == 0)
	{
		int a, b;

		if(readOvrParams(fd, &a, &b) != SB_OK)
			printf("Error reading parameters\n");
		else
			printf("a=%d, b=%d\n", a, b);
		close(fd);
		return 0;
	}

    else if(strcmp(argv[1], "-setovr") == 0)
    {
		if(argc != 4)
		{
			printf("Bad number of arguments\n");
			close(fd);
			exit(1);
		}

		printf("reply was 0x%02x\n", setOvrParams(fd, atoi(argv[2]), atoi(argv[3])));
	}

    else if(strcmp(argv[1], "-bfreset") == 0)
    {
        printf("reply was 0x%02x\n", resetBlackfin(fd));
    }

    else if(strcmp(argv[1], "-bfstart") == 0)
    {
        printf("reply was 0x%02x\n", startBlackfin(fd));
    }

    else if(strcmp(argv[1], "-bfstop") == 0)
    {
        printf("reply was 0x%02x\n", stopBlackfin(fd));
    }

    else if(strcmp(argv[1], "-sonar") == 0)
    {
        struct sonarData sd;
        getSonarData(fd, &sd);

        printf("Vector: \t<%5.4f %5.4f %5.4f>\n", sd.vectorX, sd.vectorY, sd.vectorZ);
        printf("Status: \t0x%02x\n", sd.status);
        printf("Range:  \t%u\n", sd.range);
        printf("Timestamp:\t%u\n", sd.timeStampSec);
        printf("Sample No:\t%u\n", sd.timeStampUSec);
//         printf("reply was 0x%02x\n", resetBlackfin(fd));
    }

    else if(strcmp(argv[1], "-magpwron") == 0)
    {
        int ret;
        if((ret = setMagPower(fd, 1)) != SB_OK)
            printf("Error: %s\n", sbErrorToText(ret));
    }

    else if(strcmp(argv[1], "-magpwroff") == 0)
    {
        int ret;
        if((ret = setMagPower(fd, 0)) != SB_OK)
            printf("Error: %s\n", sbErrorToText(ret));
    }
    
    //kanga 7/2/2013
    else if(strcmp(argv[1], "-camconnect") == 0)
    {
        int ret;
        if(argc != 2) {
            printf("Incorrect parameters for -camconnect!\n");
        } 
        else {
            if((ret = camConnect(fd)) != SB_OK)
                printf("Error: %s\n", sbErrorToText(ret));
        }
    }

    else if(strcmp(argv[1], "-camdisconnect") == 0)
    {
        int ret;
        if(argc != 2) {
            printf("Incorrect parameters for -camdisconnect!\n");
        } 
        else {
            if((ret = camDisconnect(fd)) != SB_OK)
                printf("Error: %s\n", sbErrorToText(ret));
        }        
    }    



    else if(strcmp(argv[1], "-setspeed") == 0)
    {
        if(argc != 8)
        {
                printf("Bad number of arguments\n");
                close(fd);
                exit(1);
        }

        printf("Ctrl-C to quit.\n");
        while(1)
        {
            setSpeeds(fd, atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), atoi(argv[6]), atoi(argv[7]));
            usleep(50000);
        }
    }
    else if(strcmp(argv[1], "-status") == 0)
    {
        int ret;
        int bStat = -1;
        int bEnable = -1;

        unsigned char temp[NUM_TEMP_SENSORS];
        ret = readDepth(fd);

        if(ret == SB_ERROR)
            printf("Error reading depth!\n");
        else
            printf("Depth :   %d (%s)\n", ret, (ret > 50) ? "Seems OK" : "Check Sensor");

        ret = readTemp(fd, temp);
        if(ret == SB_ERROR)
            printf("Error reading temperature!\n");
        else
        {
            int i;
            printf("Temperature (C): ");

            for(i=0; i<NUM_TEMP_SENSORS; i++)
                if(temp[i] == 255)
                    printf("%c%c ", '?','?');
                else
                    printf("%d ", temp[i]);
            printf("\n");
        }

        ret = readThrusterState(fd);
        if(ret == SB_ERROR)
            printf("Error reading thruster state!\n");
        else
        {
            printf("\nThruster state: 0x%02X\t", ret);
            printf("[%c%c%c%c%c%c]",
            (ret & THRUSTER1_ENABLED) ? '1' : '-',
            (ret & THRUSTER2_ENABLED) ? '2' : '-',
            (ret & THRUSTER3_ENABLED) ? '3' : '-',
            (ret & THRUSTER4_ENABLED) ? '4' : '-',
            (ret & THRUSTER5_ENABLED) ? '5' : '-',
            (ret & THRUSTER6_ENABLED) ? '6' : '-');

            if(ret == 0x00)
                printf("   (All safe)");

            if(ret == 0x3F)
                printf("   (All unsafe)");

        }

        ret = readOvrState(fd);
        if(ret == SB_ERROR)
            printf("\nError reading overcurrent register!\n");
        else
        {
            printf("\nOver-current: \t0x%02X", ret);
            printf("\t[%c%c%c%c%c%c]",
            (ret & THRUSTER1_OVR) ? '1' : '-',
            (ret & THRUSTER2_OVR) ? '2' : '-',
            (ret & THRUSTER3_OVR) ? '3' : '-',
            (ret & THRUSTER4_OVR) ? '4' : '-',
            (ret & THRUSTER5_OVR) ? '5' : '-',
            (ret & THRUSTER6_OVR) ? '6' : '-');

            if(ret == 0x00)
                printf("   (All ok)");

            if(ret == 0xFF)
                printf("   (All busted)");
        }


        ret = readBarState(fd);
        if(ret == SB_ERROR)
            printf("Error reading bar state!\n");
        else
        {
            printf("\nBar state   :\t0x%02X", ret);
            printf("\t[%c%c%c%c%c%c%c%c]",
            (ret & BAR1_ENABLED) ? '1' : '-',
            (ret & BAR2_ENABLED) ? '2' : '-',
            (ret & BAR3_ENABLED) ? '3' : '-',
            (ret & BAR4_ENABLED) ? '4' : '-',
            (ret & BAR5_ENABLED) ? '5' : '-',
            (ret & BAR6_ENABLED) ? '6' : '-',
            (ret & BAR7_ENABLED) ? '7' : '-',
            (ret & BAR8_ENABLED) ? '8' : '-');

            if(ret == 0x00)
                printf(" (All off)");

            if(ret == 0xFF)
                printf(" (All on)");
        }

        ret = readBatteryEnables(fd);
        if(ret == SB_ERROR)
            printf("Error reading battery enables state!\n");
        else
        {
            printf("\nBatt enables:   0x%02X\t", ret);
            printf("[%c%c%c%c%c%c]",
            (ret & BATT1_ENABLED) ? '1' : '-',
            (ret & BATT2_ENABLED) ? '2' : '-',
            (ret & BATT3_ENABLED) ? '3' : '-',
            (ret & BATT4_ENABLED) ? '4' : '-',
            (ret & BATT5_ENABLED) ? '5' : '-',
            (ret & BATT6_ENABLED) ? '6' : '-');

            if(ret == 0x00)
                printf("  (All off ? ? ? ? ?)");

            if(ret == 0x1F)
                printf("    (All on)");

            bEnable = ret;
        }

        ret = readStatus(fd);
        if(ret == SB_ERROR)
            printf("Error reading board status!\n");
        else
        {
            printf("\nBatt in use :   0x%02X\t", ret);
            printf("[%c%c%c%c%c%c]",
            (ret & BATT1_INUSE) ? '1' : '-',
            (ret & BATT2_INUSE) ? '2' : '-',
            (ret & BATT3_INUSE) ? '3' : '-',
            (ret & BATT4_INUSE) ? '4' : '-',
            (ret & BATT5_INUSE) ? '5' : '-',
            (ret & BATT6_INUSE) ? '6' : '-');

            bStat = ret;

            printf("\n\nStatus: 0x%02X\n", ret);
            printf("\t%s\n\t%s\n\t%s\n",
                (ret & STATUS_WATER) ? "Water present" : "No water detected.",
                (ret & STATUS_STARTSW) ? "Start switch on" : "Start switch off",
                (ret & STATUS_KILLSW) ? "Kill magnet present" : "No kill magnet");
        }

        struct powerInfo info;

        ret = readMotorCurrents(fd, &info);
        if(ret == SB_OK)
        {
            int i;
            printf("\nOutput currents:\n\t[1-4]: ");
            for(i=0; i<4; i++)
                printf("\t%2.3fA", info.motorCurrents[i]);
            printf("\n\t[5-8]: ");
            for(i=0; i<4; i++)
                printf("\t%2.3fA", info.motorCurrents[i+4]);
        } else
            printf("\nError reading motor currents\n");

        printf("\n");


        if(readBoardVoltages(fd, &info) == SB_OK &&
           readBatteryVoltages(fd, &info) == SB_OK &&
           readBatteryCurrents(fd, &info) == SB_OK)
        {
            printf("\nPower information:\n");
            printf("\t5V Bus :\t% 2.3fV\t  % 2.3fA\n", info.v5VBus, info.i5VBus);
            printf("\t12V Bus:\t%-2.3fV\t  % 2.3fA\n", info.v12VBus, info.i12VBus);
            printf("\t26V Bus:\t% 2.3fV  (Not implemented)\n", info.v26VBus);
            printf("\tAux Current:\t %2.3fA\n", info.iAux);


            if(bStat == -1 || bEnable == -1)
                printf("Could not get enough information about batteries\n");
            else
            {
                printf("\nBattery information:\n");

                unsigned char enableFlags[]=
                {
                    BATT1_ENABLED, BATT2_ENABLED, BATT3_ENABLED, BATT4_ENABLED, BATT5_ENABLED, BATT6_ENABLED
                };

                unsigned char useFlags[]=
                {
                    BATT1_INUSE, BATT2_INUSE, BATT3_INUSE, BATT4_INUSE, BATT5_INUSE, BATT6_INUSE
                };

                int i;

                printf("\tBAT\t Voltage\t Current   EN?\tUsed?\n");
                for(i=0; i<6; i++)
                {
                    printf("\t");
                    if(i != 4)
                        printf("B%d ", i+1);
                    else
                        printf("EXT ");

                    printf("\t% 2.3fV\t% 2.3fA\t ", info.battVoltages[i], info.battCurrents[i]);

                    printf("   %c\t %c", bEnable & enableFlags[i] ? 'Y' : 'N',
                                     bStat & useFlags[i] ? 'Y' : 'N');


                    printf("\n");
                }
            }


        } else
            printf("\nError reading power information\n");




    }


    else if(strcmp(argv[1], "-check") == 0)
    {
        int ret = pingBoard(fd);

        if(ret == SB_OK)
            printf("Ping: OK\n");
        else
            printf("Ping: Error (Code %d)\n", ret);

        ret = checkBoard(fd);

        if(ret == SB_OK)
            printf("Self-test: OK\n");
        else
            printf("Self-test: Error (Code %d)\n", ret);

        close(fd);
        return 0;
    }

    else if(strcmp(argv[1], "-marker") == 0)
    {
        if(argc == 3)
        {
            int t = atoi(argv[2]);
            if(t != 1 && t != 2)
            {
                printf("Bad marker number: %d\n", t);
                close(fd);
                return -1;
            }

            dropMarker(fd, t-1);

        } else
        {
            printf("Which marker? Specify 1 or 2.\n");
        }

        close(fd);
        return 0;
    }

    else if(strcmp(argv[1], "-safe") == 0)
    {
        int i;
        unsigned int cmdList[]=
        {
            CMD_THRUSTER1_OFF, CMD_THRUSTER2_OFF, CMD_THRUSTER3_OFF,
            CMD_THRUSTER4_OFF, CMD_THRUSTER5_OFF, CMD_THRUSTER6_OFF
        };

        if(argc == 3)
        {
            int t = atoi(argv[2]);
            if(t < 1 || t > 6)
            {
                printf("Bad thruster number: %d\n", t);
                close(fd);
                return -1;
            }

            thrusterCmd(fd, cmdList[t-1]);

        } else
        {
            for(i=0; i<6; i++)
                thrusterCmd(fd, cmdList[i]);
        }

        close(fd);
        return 0;
    }

    else if(strcmp(argv[1], "-battoff") == 0)
    {
        unsigned int cmdList[]=
        {
            CMD_BATT1_OFF, CMD_BATT2_OFF, CMD_BATT3_OFF, CMD_BATT4_OFF, CMD_BATT5_OFF
        };

        if(argc == 3)
        {
            int t = atoi(argv[2]);
            if(t < 1 || t > 5)
            {
                printf("Bad battery number: %d\n", t);
                close(fd);
                return -1;
            }

            setBatteryState(fd, cmdList[t-1]);

        } else
        {
            printf("Battery number not specified.\n");
        }

        close(fd);
        return 0;
    }

    else if(strcmp(argv[1], "-batton") == 0)
    {
        unsigned int cmdList[]=
        {
            CMD_BATT1_ON, CMD_BATT2_ON, CMD_BATT3_ON, CMD_BATT4_ON, CMD_BATT5_ON
        };

        if(argc == 3)
        {
            int t = atoi(argv[2]);
            if(t < 1 || t > 5)
            {
                printf("Bad battery number: %d\n", t);
                close(fd);
                return -1;
            }

            setBatteryState(fd, cmdList[t-1]);

        } else
        {
            printf("Battery number not specified.\n");
        }

        close(fd);
        return 0;
    }

    else if(strcmp(argv[1], "-unsafe") == 0)
    {
        int i;
        unsigned int cmdList[]=
        {
            CMD_THRUSTER1_ON, CMD_THRUSTER2_ON, CMD_THRUSTER3_ON,
            CMD_THRUSTER4_ON, CMD_THRUSTER5_ON, CMD_THRUSTER6_ON
        };

        if(argc == 3)
        {
            int t = atoi(argv[2]);
            if(t < 1 || t > 6)
            {
                printf("Bad thruster number: %d\n", t);
                close(fd);
                return -1;
            }

            thrusterCmd(fd, cmdList[t-1]);

        } else
        {
            for(i=0; i<6; i++)
                thrusterCmd(fd, cmdList[i]);
        }


        close(fd);
        return 0;
    }



    else if(strcmp(argv[1], "-baroff") == 0)
    {
        int i;
        unsigned int cmdList[]=
        {
            CMD_BAR1_OFF, CMD_BAR2_OFF, CMD_BAR3_OFF, CMD_BAR4_OFF,
            CMD_BAR5_OFF, CMD_BAR6_OFF, CMD_BAR7_OFF, CMD_BAR8_OFF
        };

        if(argc == 3)
        {
            int t = atoi(argv[2]);
            if(t < 1 || t > 8)
            {
                printf("Bad bar number: %d\n", t);
                close(fd);
                return -1;
            }

            barCmd(fd, cmdList[t-1]);

        } else
        {
            for(i=0; i<8; i++)
                barCmd(fd, cmdList[i]);
        }

        close(fd);
        return 0;
    }



    else if(strcmp(argv[1], "-baron") == 0)
    {
        int i;
        unsigned int cmdList[]=
        {
            CMD_BAR1_ON, CMD_BAR2_ON, CMD_BAR3_ON, CMD_BAR4_ON,
            CMD_BAR5_ON, CMD_BAR6_ON, CMD_BAR7_ON, CMD_BAR8_ON
        };

        if(argc == 3)
        {
            int t = atoi(argv[2]);
            if(t < 1 || t > 8)
            {
                printf("Bad bar number: %d\n", t);
                close(fd);
                return -1;
            }

            barCmd(fd, cmdList[t-1]);

        } else
        {
            for(i=0; i<8; i++)
                barCmd(fd, cmdList[i]);
        }

        close(fd);
        return 0;
    }


    else if(strcmp(argv[1], "-setbars") == 0)
    {
    	if(argc == 1)
	{
	    printf("Segmentation fault\n");
	    close(fd);
	    return -1;
	}

    	int t = atoi(argv[2]);

        setBarOutputs(fd, t);

        close(fd);
        return 0;
    }


    else if(strcmp(argv[1], "-noblink") == 0)
    {
        setAnimation(fd, ANIMATION_NONE);
        close(fd);
        return 0;
    }


    else if(strcmp(argv[1], "-redblue") == 0)
    {
        setAnimation(fd, ANIMATION_REDBLUE);
        close(fd);
        return 0;
    }

    else if(strcmp(argv[1], "-redgreen") == 0)
    {
        setAnimation(fd, ANIMATION_REDGREEN);
        close(fd);
        return 0;
    }

    else if(strcmp(argv[1], "-tstop") == 0)
    {
        setSpeeds(fd, 0, 0, 0, 0, 0, 0);
        close(fd);
        return 0;
    }

    else if(strcmp(argv[1], "-hardkill") == 0)
    {
        hardKill(fd);
        close(fd);
        return 0;
    }


    else if(strcmp(argv[1], "-diagon") == 0)
    {
        setDiagnostics(fd, 1);
        close(fd);
        return 0;
    }

    else if(strcmp(argv[1], "-diagoff") == 0)
    {
        setDiagnostics(fd, 0);
        close(fd);
        return 0;
    }


    else if(strcmp(argv[1], "-bloff") == 0)
    {
        lcdBacklight(fd, LCD_BL_OFF);
        close(fd);
        return 0;
    }

    else if(strcmp(argv[1], "-blon") == 0)
    {
        lcdBacklight(fd, LCD_BL_ON);
        close(fd);
        return 0;
    }

    else if(strcmp(argv[1], "-blfl") == 0)
    {
        lcdBacklight(fd, LCD_BL_FLASH);
        close(fd);
        return 0;
    }

    else if(strcmp(argv[1], "-intpower") == 0)
    {
        switchToInternalPower(fd);
        close(fd);
        return 0;
    }

    else if(strcmp(argv[1], "-extpower") == 0)
    {
        switchToExternalPower(fd);
        close(fd);
        return 0;
    }

    else if(strcmp(argv[1], "-c") == 0)
    {
        displayText(fd, 0, "");
        displayText(fd, 1, "");
        close(fd);
        return 0;
    }



    else if(strcmp(argv[1], "-s") == 0)
    {
        displayText(fd, 0, "Please attach   ");
        displayText(fd, 1, "Start magnet    ");

        while((readStatus(fd) & STATUS_STARTSW) == 0)
        {
            usleep(100*1000);
        }

        displayText(fd, 0, "Ready to start  ");
    	displayText(fd, 1, "");

    	while((readStatus(fd) & STATUS_STARTSW))
        {
            usleep(100*1000);
        }
        displayText(fd, 0, "Running...");
        close(fd);
        return 0;
    }


    else if(strcmp(argv[1], "-t") == 0)
    {
        displayText(fd, 0, argv[2]);

        if(argc == 4)
            displayText(fd, 1, argv[3]);
    }

    else if(strcmp(argv[1], "-dvlon") == 0)
    {
        int ret;
        if((ret = DVLOn(fd, 1)) != SB_OK)
            printf("Error: %s\n", sbErrorToText(ret));
    }

    else if(strcmp(argv[1], "-dvloff") == 0)
    {
        int ret;
        if((ret = DVLOn(fd, 0)) != SB_OK)
            printf("Error: %s\n", sbErrorToText(ret));
    }

    else if(strcmp(argv[1], "-magpwron") == 0)
    {
        int ret;
        if((ret = setMagPower(fd, 1)) != SB_OK)
            printf("Error: %s\n", sbErrorToText(ret));
    }

    else if(strcmp(argv[1], "-magpwroff") == 0)
    {
        int ret;
        if((ret = setMagPower(fd, 0)) != SB_OK)
            printf("Error: %s\n", sbErrorToText(ret));
    }

    else if(strcmp(argv[1], "-firetorp") == 0)
    {
        int ret;
        if(argc != 3) {
            printf("I need a torpedo number!\n");
        } else if((ret = fireTorpedo(fd, atoi(argv[2]))) != SB_OK)
            printf("Error: %s\n", sbErrorToText(ret));
    }

    else if(strcmp(argv[1], "-armtorp") == 0)
    {
        int ret;
        if(argc != 3) {
            printf("I need a torpedo number!\n");
        }
        else if((ret = armTorpedo(fd, atoi(argv[2]))) != SB_OK)
            printf("Error: %s\n", sbErrorToText(ret));
    }

    else if(strcmp(argv[1], "-voidtorp") == 0)
    {
        int ret;
        if(argc != 3) {
            printf("I need a torpedo number!\n");
        }
        else if((ret = voidTorpedo(fd, atoi(argv[2]))) != SB_OK)
            printf("Error: %s\n", sbErrorToText(ret));
    }

    else if(strcmp(argv[1], "-extgrabber") == 0)
    {
        int ret;
        if(argc == 2)
        {
            if((ret = extendGrabber(fd, 0)) != SB_OK)
            printf("Error: %s\n", sbErrorToText(ret));
        }
        else if (argc == 3 && (argv[1]<=2 && argv[1]>=0)) 
        {
            if((ret = extendGrabber(fd, argv[2])) != SB_OK)
            printf("Error: %s\n", sbErrorToText(ret));
        }
        else printf("Error: Return 0 or nothing for BOTH, 1 for Grabber 1, and 2 for Grabber 2");
    }

    else if(strcmp(argv[1], "-retgrabber") == 0)
    {
        int ret;
        if((ret = retractGrabber(fd)) != SB_OK)
            printf("Error: %s\n", sbErrorToText(ret));
    }

    else if(strcmp(argv[1], "-voidgrabber") == 0)
    {
        int ret;
        if((ret = voidGrabber(fd)) != SB_OK)
            printf("Error: %s\n", sbErrorToText(ret));
    }

    else if(strcmp(argv[1], "-voidall") == 0)
    {
        int ret;
        if((ret = voidSystem(fd)) != SB_OK)
            printf("Error: %s\n", sbErrorToText(ret));
    }

    else if(strcmp(argv[1], "-pneuoff") == 0)
    {
        int ret;
        if((ret = pneumaticsOff(fd)) != SB_OK)
            printf("Error: %s\n", sbErrorToText(ret));
    }

    else if(strcmp(argv[1], "-derpypwr") == 0)
    {
        int ret;

        if(argc != 3) {
            printf("No parameter for -derpypwr!\n");
        } else if(strcmp(argv[2], "on") == 0) {
            if((ret = setDerpyPower(fd, 1)) != SB_OK)
                printf("Error: %s\n", sbErrorToText(ret));
        } else if(strcmp(argv[2], "off") == 0) {
            if((ret = setDerpyPower(fd, 0)) != SB_OK)
                printf("Error: %s\n", sbErrorToText(ret));
        } else {
            printf("Invalid parameter for -derpypwr!\n");
        }
    }
    
    else if(strcmp(argv[1], "-setderpy") == 0)
    {
        int ret;
        if(argc != 3) {
            printf("No parameter for -setderpy!\n");
        } else {
            if((ret = setDerpySpeed(fd, atoi(argv[2]))) != SB_OK)
                printf("Error: %s\n", sbErrorToText(ret));
        }
    }

    else if(strcmp(argv[1], "-stopderpy") == 0)
    {
        int ret;
        if((ret = stopDerpy(fd)) != SB_OK)
            printf("Error: %s\n", sbErrorToText(ret));
    }
    
    else {
        printf("error: too much bullshit: %s\n", argv[1]);
        return -1;
    }

    close(fd);
    return 0;
}

