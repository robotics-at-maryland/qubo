SensHub Internet of Things Example

This application uses FreeRTOS to manage multiple sensor tasks and
aggregate sensor data to be published to a cloud server. The senshub_iot.c
file contains the main function and perform task init before handing
control over to the FreeRTOS scheduler.

The tasks and their responsibilities are as follows:

- cloud_task.c is a manager of the cloud interface.  It gathers the sensor
  data and builds it into a packet for transmission to the cloud.

- command_task.c is a manager of the UART virtual com port connection to a
  local PC.  This interface allows advanced commands and data.

- isl29023_task.c is a task to manage the interface to the isl29023 light
  sensor.  It collects data from the sensor and makes it available to
  other tasks.

- tmp006_task.c is a task that manages the tmp006 temperature sensor. It
  gathers data from the temperature sensor and makes it available to other
  tasks.

- bmp180_task.c is a task that manages the bmp180 pressure sensor. It
  gathers data from the sensor and makes it available to other tasks.

- compdcm_task.c is a task that manages data from the MPU9150. It performs
  complimentary direct cosine matrix filter on the data to determine roll,
  pitch and yaw as well as quaternions. This data is made available to
  other tasks.

- sht21_task.c is a task that manages the SHT21 humidity and temperature
  sensor.  It collects data from the sensor and makes it available to other
  tasks.

In addition to the tasks, this application also uses the following FreeRTOS
resources:

- Queues enable information transfer between tasks.

- Mutex Semaphores guard resources such as the UART from access by multiple
  tasks at the same time.

- Binary Semaphores synchronize events between interrupts and task contexts.

- A FreeRTOS Delay to put the tasks in blocked state when they have nothing
  to do.

- A Software timer to regulate the timing of cloud sync events.

- The FreeRTOS run time stats feature to show CPU usage of each task at run
  time.

For additional details on FreeRTOS, refer to the FreeRTOS web page at:
http://www.freertos.org/

-------------------------------------------------------------------------------

Copyright (c) 2013-2016 Texas Instruments Incorporated.  All rights reserved.
Software License Agreement

Texas Instruments (TI) is supplying this software for use solely and
exclusively on TI's microcontroller products. The software is owned by
TI and/or its suppliers, and is protected under applicable copyright
laws. You may not combine this software with "viral" open-source
software in order to form a larger program.

THIS SOFTWARE IS PROVIDED "AS IS" AND WITH ALL FAULTS.
NO WARRANTIES, WHETHER EXPRESS, IMPLIED OR STATUTORY, INCLUDING, BUT
NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE APPLY TO THIS SOFTWARE. TI SHALL NOT, UNDER ANY
CIRCUMSTANCES, BE LIABLE FOR SPECIAL, INCIDENTAL, OR CONSEQUENTIAL
DAMAGES, FOR ANY REASON WHATSOEVER.

This is part of revision 2.1.3.156 of the EK-TM4C1294XL Firmware Package.
