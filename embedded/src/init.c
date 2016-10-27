/*
Copyright 2014, Jernej Kovacic

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

/**
 * @file
 *
 * The file contains a function that initializes the
 * MCU's peripherals.
 *
 * @author Jernej Kovacic
 */


#include "FreeRTOSConfig.h"

#include "sysctl.h"
#include "fpu.h"

/*
 * Implemented in gpio.c, sysctl.c and watchdog.c, respectively
 */
extern void _gpio_initIntHandlers(void);
extern void _sysctl_enableGpioAhb(void);
extern void _wd_initIntHandlers(void);


/*
 * An "unofficial" function (not exposed in any header)
 * that performs initialization of MCU's peripherals.
 *
 * It should only be called from the startup routine before
 * the execution is passed to a user application
 * (typically started in main().)
 */
void _init(void)
{
    /* Initializes the MCU revision number: */
    sysctl_mcuRevision();

    /* Configure system clock frequency to 50 MHz (default) */
    sysctl_configSysClock(APP_SYS_CLOCK_DIV);

    /* Depending on configuration, enable GPIO AHB mode: */
    if ( 0 != APP_GPIO_AHB )
    {
        _sysctl_enableGpioAhb();
    }


    /* Enable/disable FPU: */
    if ( 0 != APP_FPU_ENABLE )
    {
        fpu_enable();

        /* Enable/disable lazy stacking of FPU's registers */
        if ( 0 != APP_FPU_LAZY_STACKING )
        {
        	fpu_enableLazyStacking();
        }
        else
        {
            fpu_enableStacking();
        }
    }
    else
    {
        fpu_disable();
    }

    /*
     * Initialize the tables of GPIO and
     * watchdog interrupt handlers.
     */
    _gpio_initIntHandlers();
    _wd_initIntHandlers();
}
