//*****************************************************************************
//
// senshub_iot.c - Example to publish SensorHub BoosterPack data to the cloud.
//
// Copyright (c) 2013-2016 Texas Instruments Incorporated.  All rights reserved.
// Software License Agreement
// 
// Texas Instruments (TI) is supplying this software for use solely and
// exclusively on TI's microcontroller products. The software is owned by
// TI and/or its suppliers, and is protected under applicable copyright
// laws. You may not combine this software with "viral" open-source
// software in order to form a larger program.
// 
// THIS SOFTWARE IS PROVIDED "AS IS" AND WITH ALL FAULTS.
// NO WARRANTIES, WHETHER EXPRESS, IMPLIED OR STATUTORY, INCLUDING, BUT
// NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE APPLY TO THIS SOFTWARE. TI SHALL NOT, UNDER ANY
// CIRCUMSTANCES, BE LIABLE FOR SPECIAL, INCIDENTAL, OR CONSEQUENTIAL
// DAMAGES, FOR ANY REASON WHATSOEVER.
// 
// This is part of revision 2.1.3.156 of the EK-TM4C1294XL Firmware Package.
//
//*****************************************************************************

#include <stdbool.h>
#include <stdint.h>
#include "inc/hw_memmap.h"
#include "inc/hw_types.h"
#include "inc/hw_ints.h"
#include "driverlib/gpio.h"
#include "driverlib/pin_map.h"
#include "driverlib/rom.h"
#include "driverlib/rom_map.h"
#include "driverlib/sysctl.h"
#include "driverlib/interrupt.h"
#include "driverlib/timer.h"
#include "sensorlib/i2cm_drv.h"
#include "utils/uartstdio.h"
#include "FreeRTOS.h"
#include "task.h"
#include "queue.h"
#include "semphr.h"
#include "tmp006_task.h"
#include "bmp180_task.h"
#include "isl29023_task.h"
#include "compdcm_task.h"
#include "sht21_task.h"
#include "command_task.h"
#include "cloud_task.h"
#include "drivers/pinout.h"
#include "drivers/buttons.h"

//*****************************************************************************
//
//! \addtogroup example_list
//! <h1>SensHub Internet of Things Example (senshub_iot)</h1>
//!
//! This application uses FreeRTOS to manage multiple sensor tasks and
//! aggregate sensor data to be published to a cloud server. The senshub_iot.c
//! file contains the main function and perform task init before handing
//! control over to the FreeRTOS scheduler.
//!
//! The tasks and their responsibilities are as follows:
//!
//! - cloud_task.c is a manager of the cloud interface.  It gathers the sensor
//!   data and builds it into a packet for transmission to the cloud.
//!
//! - command_task.c is a manager of the UART virtual com port connection to a
//!   local PC.  This interface allows advanced commands and data.
//!
//! - isl29023_task.c is a task to manage the interface to the isl29023 light
//!   sensor.  It collects data from the sensor and makes it available to
//!   other tasks.
//!
//! - tmp006_task.c is a task that manages the tmp006 temperature sensor. It
//!   gathers data from the temperature sensor and makes it available to other
//!   tasks.
//!
//! - bmp180_task.c is a task that manages the bmp180 pressure sensor. It
//!   gathers data from the sensor and makes it available to other tasks.
//!
//! - compdcm_task.c is a task that manages data from the MPU9150. It performs
//!   complimentary direct cosine matrix filter on the data to determine roll,
//!   pitch and yaw as well as quaternions. This data is made available to
//!   other tasks.
//!
//! - sht21_task.c is a task that manages the SHT21 humidity and temperature
//!   sensor.  It collects data from the sensor and makes it available to other
//!   tasks.
//!
//! In addition to the tasks, this application also uses the following FreeRTOS
//! resources:
//!
//! - Queues enable information transfer between tasks.
//!
//! - Mutex Semaphores guard resources such as the UART from access by multiple
//!   tasks at the same time.
//!
//! - Binary Semaphores synchronize events between interrupts and task contexts.
//!
//! - A FreeRTOS Delay to put the tasks in blocked state when they have nothing
//!   to do.
//!
//! - A Software timer to regulate the timing of cloud sync events.
//!
//! - The FreeRTOS run time stats feature to show CPU usage of each task at run
//!   time.
//!
//! For additional details on FreeRTOS, refer to the FreeRTOS web page at:
//! http://www.freertos.org/
//
//*****************************************************************************


//*****************************************************************************
//
// Global variable to hold the system clock speed.
//
//*****************************************************************************
uint32_t g_ui32SysClock;

//*****************************************************************************
//
// The mutex that protects concurrent access of I2C from multiple tasks.
//
//*****************************************************************************
xSemaphoreHandle g_xI2CSemaphore;

//*****************************************************************************
//
// The mutex that protects concurrent access of Cloud Data from multiple
// tasks.
//
//*****************************************************************************
xSemaphoreHandle g_xCloudDataSemaphore;

//*****************************************************************************
//
// The instance structure that will hold data for the cloud from all the
// sensors.
//
//*****************************************************************************
sCloudData_t g_sCloudData;

//*****************************************************************************
//
// Global instance structure for the I2C master driver.
//
//*****************************************************************************
tI2CMInstance g_sI2CInst;

//*****************************************************************************
//
// Counter value used by the FreeRTOS run time stats feature.
// http://www.freertos.org/rtos-run-time-stats.html
//
//*****************************************************************************
volatile unsigned long g_vulRunTimeStatsCountValue;

//*****************************************************************************
//
// The error routine that is called if the driver library encounters an error.
//
//*****************************************************************************
#ifdef DEBUG
void
__error__(char *pcFilename, uint32_t ui32Line)
{
}

#endif

//*****************************************************************************
//
// Interrupt handler for Timer0A.
//
// This function will be called periodically on the expiration of Timer0A It
// performs periodic tasks, such as looking for input on the physical buttons,
// and reporting usage statistics to the cloud.
//
//*****************************************************************************
void
Timer0IntHandler(void)
{
    //
    // Clear the timer interrupt.
    //
    ROM_TimerIntClear(TIMER0_BASE, TIMER_TIMA_TIMEOUT);

    //
    // Keep track of the number of times this interrupt handler has been
    // called.
    //
    g_vulRunTimeStatsCountValue++;

}

//*****************************************************************************
//
// Configure and start the timer that will increment the variable used to
// track FreeRTOS task statistics.
//
//*****************************************************************************
void SensorCloudStatTimerConfig(void)
{
    //
    // Enable the peripherals used by this example.
    //
    ROM_SysCtlPeripheralEnable(SYSCTL_PERIPH_TIMER0);

    //
    // Configure the two 32-bit periodic timers.  The period of the timer for
    // FreeRTOS run time stats must be at least 10 times faster than the tick
    // rate.
    //
    ROM_TimerConfigure(TIMER0_BASE, TIMER_CFG_PERIODIC);
    ROM_TimerLoadSet(TIMER0_BASE, TIMER_A, g_ui32SysClock /
                                           (configTICK_RATE_HZ * 10));

    //
    // Setup the interrupts for the timer timeouts.
    //
    ROM_IntEnable(INT_TIMER0A);
    ROM_TimerIntEnable(TIMER0_BASE, TIMER_TIMA_TIMEOUT);

    TimerEnable(TIMER0_BASE, TIMER_A);

}

//*****************************************************************************
//
// Called by the NVIC as a result of I2C Interrupt. I2C7 is the I2C connection
// to the Senshub BoosterPack for BoosterPack 1 interface.  I2C8 must be used
// for BoosterPack 2 interface.  Must also move this function pointer in the
// startup file interrupt vector table for your tool chain if using BoosterPack
// 2 interface headers.
//
//*****************************************************************************
void
SensHubI2CIntHandler(void)
{
    //
    // Pass through to the I2CM interrupt handler provided by sensor library.
    // This is required to be at application level so that I2CMIntHandler can
    // receive the instance structure pointer as an argument.
    //
    I2CMIntHandler(&g_sI2CInst);
}

//*****************************************************************************
//
// This hook is called by FreeRTOS when an stack overflow error is detected.
//
//*****************************************************************************
void
vApplicationStackOverflowHook(xTaskHandle *pxTask, char *pcTaskName)
{
    //
    // This function can not return, so loop forever.  Interrupts are disabled
    // on entry to this function, so no processor interrupts will interrupt
    // this loop.
    //
    while(1)
    {
    }
}

//*****************************************************************************
//
// Initialize FreeRTOS and start the initial set of tasks.
//
//*****************************************************************************
int
main(void)
{
    //
    // Configure the system frequency.
    //
    g_ui32SysClock = MAP_SysCtlClockFreqSet((SYSCTL_XTAL_25MHZ |
                                             SYSCTL_OSC_MAIN |
                                             SYSCTL_USE_PLL |
                                             SYSCTL_CFG_VCO_480), 120000000);

    //
    // Configure the device pins for this board.
    // This application uses Ethernet but not USB.
    //
    PinoutSet(true, false);
    ButtonsInit();

    //
    // The I2C7 peripheral must be enabled before use.
    //
    // For BoosterPack 2 interface use I2C8 and GPIOA.
    //
    ROM_SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOD);
    ROM_SysCtlPeripheralEnable(SYSCTL_PERIPH_I2C7);

    //
    // Configure the pin muxing for I2C7 functions on port D0 and D1.
    // This step is not necessary if your part does not support pin muxing.
    //
    // For BoosterPack 2 interface use PA2 and PA3.
    //
    ROM_GPIOPinConfigure(GPIO_PD0_I2C7SCL);
    ROM_GPIOPinConfigure(GPIO_PD1_I2C7SDA);

    //
    // Select the I2C function for these pins.  This function will also
    // configure the GPIO pins pins for I2C operation, setting them to
    // open-drain operation with weak pull-ups.  Consult the data sheet
    // to see which functions are allocated per pin.
    //
    // For BoosterPack 2 interface use PA2 and PA3.
    //
    GPIOPinTypeI2CSCL(GPIO_PORTD_BASE, GPIO_PIN_0);
    ROM_GPIOPinTypeI2C(GPIO_PORTD_BASE, GPIO_PIN_1);

    //
    // Initialize I2C peripheral driver.
    //
    // For BoosterPack 2 interface use I2C8
    //
    I2CMInit(&g_sI2CInst, I2C7_BASE, INT_I2C7, 0xff, 0xff, g_ui32SysClock);
    IntPrioritySet(INT_I2C7, 0xE0);

    //
    // Create a mutex to guard the I2C.
    //
    g_xI2CSemaphore = xSemaphoreCreateMutex();

    //
    // Create a mutex to guard the Cloud Data structure.
    //
    g_xCloudDataSemaphore = xSemaphoreCreateMutex();

    //
    // Create the virtual com port task.
    // Doing this task first initializes the UART.
    //
    if(CommandTaskInit() != 0)
    {
        //
        // Init returned an error. Print an alert to the user and
        // spin forever.  Wait for reset or user to debug.
        //
        UARTprintf("Virtual COM Port: Task Init Failed!");
        while(1)
        {
            //
            // Do Nothing.
            //
        }
    }

    //
    // Create the cloud task.
    //
    if(CloudTaskInit() != 0)
    {
        //
        // Print an error message.
        //
        UARTprintf("CloudTask: Init Failed!\n");

        while(1)
        {
            //
            // Do Nothing.
            //
        }
    }

    //
    // Create the TMP006 temperature sensor task.
    //
    if(TMP006TaskInit() != 0)
    {
        //
        // Init returned an error. Print an alert to the user and
        // spin forever.  Wait for reset or user to debug.
        //
        UARTprintf("TMP006: Task Init Failed!\n");
        while(1)
        {
            //
            // Do Nothing.
            //
        }
    }

    //
    // Create the SHT21 sensor task.
    //
    if(SHT21TaskInit() != 0)
    {
        //
        // Init returned an error. Print an alert to the user and
        // spin forever.  Wait for reset or user to debug.
        //
        UARTprintf("SHT21: Task Init Failed!\n");
        while(1)
        {
            //
            // Do Nothing.
            //
        }
    }

    //
    // Create the CompDCM 9 axis sensor task.
    //
    if(CompDCMTaskInit() != 0)
    {
        //
        // Init returned an error. Print an alert to the user and
        // spin forever.  Wait for reset or user to debug.
        //
        UARTprintf("CompDCM: Task Init Failed!\n");
        while(1)
        {
            //
            // Do Nothing.
            //
        }
    }
    //
    // Create the BMP180 sensor task.
    //
    if(BMP180TaskInit() != 0)
    {
        //
        // Init returned an error. Print an alert to the user and
        // spin forever.  Wait for reset or user to debug.
        //
        UARTprintf("BMP180: Task Init Failed!\n");
        while(1)
        {
            //
            // Do Nothing.
            //
        }
    }

    //
    // Create the ISL29023 sensor task.
    //
    if(ISL29023TaskInit() != 0)
    {
        //
        // Init returned an error. Print an alert to the user and
        // spin forever.  Wait for reset or user to debug.
        //
        UARTprintf("ISL29023: Task Init Failed!\n");
        while(1)
        {
            //
            // Do Nothing.
            //
        }
    }

    //
    // Verify that the semaphores were created correctly.
    //
    if((g_xI2CSemaphore == NULL) || (g_xCloudDataSemaphore == NULL))
    {
        //
        // I2C or CloudData semaphore was not created successfully.
        // Print an error message and wait for user to debug or reset.
        //
        UARTprintf("I2C or Cloud Data semaphore create failed.\n");
        UARTprintf("I2C Semaphore: 0x%X\t\tCloudData Semaphore: 0x%X",
                   (uint32_t) g_xI2CSemaphore,
                   (uint32_t) g_xCloudDataSemaphore);
        while(1)
        {
            //
            // Do Nothing.
            //
        }
    }

    //
    // Config and start the timer that is used by FreeRTOS to determine
    // run time stats.
    //
    SensorCloudStatTimerConfig();

    //
    // Clear the terminal and print demo introduction. This is safe here
    // since we have not yet started the scheduler.  UART and UARTStdio
    // config happens in the VCP Task. Once scheduler starts tasks must take
    // the UART semaphore to safely print.
    //
    UARTprintf("Welcome to the EK-TM4C1294XL Exosite Senshub IoT Demo "
               "using FreeRTOS!\n");

    //
    // Start the scheduler.  This should not return.
    //
    vTaskStartScheduler();

    //
    // In case the scheduler returns for some reason, print an error and loop
    // forever.
    //
    UARTprintf("RTOS scheduler returned unexpectedly.\n");
    while(1)
    {
        //
        // Do Nothing.
        //
    }
}
