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
 * Implementation of functions that manipulate both switches of
 * the Texas Instruments TM4C123GLX Launchpad directly,
 * without using general GPIO functions. This is an intermediate
 * layer, between a developer and appropriate GPIO handling functions
 * and no knowledge of switches' technical details is required.
 *
 * If additional switches are introduced, e.g. connected to additional
 * GPIO pins, it should be easy to expand the functions.
 *
 * More details about the Launchpad, see
 * Tiva(TM) C Series TM4C123G LaunchPad Evaluation Board User's Guide
 * available at:
 * http://www.ti.com/lit/ug/spmu296/spmu296.pdf
 *
 * @author Jernej Kovacic
 */


#include <stdint.h>

#include "switch.h"
#include "gpio.h"


/*
 * As evident from the page 9 of the User's Guide, the LEDs
 * are connected to the following GPIO pins:
 * - PF4: switch 1
 * - PF0: switch 2
 */

/* The GPIO port, both switches are connected to: */
#define PORT_SWITCH      ( GPIO_PORTF )

/* Pin numbers of both switches: */
#define SWITCH1_PIN      ( 4 )
#define SWITCH2_PIN      ( 0 )

/* Bit mask of pins connected to both switches: */
#define SWITCH_MASK      ( SWITCH1 | SWITCH2 )

/*
 * A value indicating it was not possible to map
 * a switch to its corresponding GPIO pin:
 * */
#define PIN_UNKNOWN      ( 255 )

/*
 * A convenience inline function that maps a switch number
 * to its corresponding pin.
 *
 * 0 is returned if 'sw' is invalid, i.e. different from 1 or 2.
 *
 * @param sw - switch number to be mapped (either 1 or 2)
 *
 * @return corresponding pin number of the provided 'sw' or 0
 */
static inline uint8_t __sw2pin(uint8_t sw)
{
    uint8_t pin = PIN_UNKNOWN;

    switch (sw)
    {
    case 1:
        pin = SWITCH1_PIN;
        break;

    case 2:
        pin = SWITCH2_PIN;
        break;
    }

    return pin;
}


/**
 * Configures both pins, connected to switches.
 *
 * Both pins are configured as regular GPIO digital
 * output pins. Additionally weak pull-up resistors
 * are applied to both pins.
 */
void switch_config(void)
{
    /*
     * Both pins are configured as regular GPIO output pins.
     */
    gpio_configGpioPin(PORT_SWITCH, SWITCH1_PIN, 0);
    gpio_configGpioPin(PORT_SWITCH, SWITCH2_PIN, 0);


    /*
     * Evident facts from the schematics at page 20 of
     * the User's Guide:
     * - When a switch is pressed, its pin's potential will
     *   equal GND, detected as logical input of 0
     * - When a switch is not pressed, its pin's potential
     *   is undefined. For that reason, pull-up resistors will
     *   be enabled for both pins. So, when a switch is not
     *   pressed, its potential will equal Vcc, detected as
     *   logical input of 1.
     */
    gpio_enablePullUp(PORT_SWITCH, SWITCH1_PIN);
    gpio_enablePullUp(PORT_SWITCH, SWITCH2_PIN);

    /*
     * Both switches will trigger interrupt requests when
     * pressed, i.e. on falling edge:
     */
    gpio_configInterrupt(PORT_SWITCH, SWITCH1_PIN, GPIOINT_EDGE_FALLING);
    gpio_configInterrupt(PORT_SWITCH, SWITCH2_PIN, GPIOINT_EDGE_FALLING);
}


/*
 * A convenience inline function that returns status of
 * GPIO inputs as specified by the input bit mask 'sw'.
 * For each switch, its corresponding bit of the return
 * value must be checked. All other bits will be cleared
 * to 0. Any set bits of 'sw' that do not belong to a valid
 * switch, will be discarded.
 *
 * @param sw - bit mask of switches of interest (SWITCH1, SWITCH2, or combination of both)
 *
 * @return bitwise statuses of all switches specified by 'sw' (1 if pressed, 0 if not)
 */
static inline uint8_t __swStatus(uint8_t sw)
{
    /* Discard all bits that do not belong to valid switches */
    const uint8_t sp = sw & SWITCH_MASK;

    /*
     * Return status of specified switches' bits must be reversed
     * and all non relevant bits must be equal to 0.
     * This can be accomplished by bitwise XORing of relevant bits
     * by 1 (reversing their values) and by additional bitwise AND
     * to filter out statuses of all irrelevant pins.
     */
    return ( (gpio_readPins(PORT_SWITCH, sp) ^ sp) & sp );

}


/**
 * @return the current status of the switch SW1 (1 if pressed, 0 if not)
 */
uint8_t switch_statusSw1(void)
{
    return
        ( 0==__swStatus(SWITCH1) ? 0 : 1 );
}


/**
 * @return the current status of the switch SW2 (1 if pressed, 0 if not)
 */
uint8_t switch_statusSw2(void)
{
    return
        ( 0==__swStatus(SWITCH2) ? 0 : 1 );
}


/**
 * @return bit mask with the current statuses of both valid switches
 *
 * @note To obtain the status of a single switch, bitwise AND the
 *       result by the switch's mask and check its corresponding bit.
 */
uint8_t switch_statusBoth(void)
{
    return ( __swStatus(SWITCH_MASK) );
}


/**
 * Enables interrupt requests, triggered on falling
 * edge of the selected switch. Additionally enables
 * the NVIC to process interrupt requests, triggered
 * pins of the switch's GPIO port. Any pending interrupts
 * from that pin are cleared.
 *
 * Nothing is done if 'sw' is invalid, i.e. different
 * from 1 or 2
 *
 * @param sw - desired switch (either 1 or 2)
 */
void switch_enableSwInt(uint8_t sw)
{
    const uint8_t pin =__sw2pin(sw);

    if ( PIN_UNKNOWN != pin )
    {
        gpio_enableInterruptPort(PORT_SWITCH);
        gpio_clearInterrupt(PORT_SWITCH, pin);
        gpio_unmaskInterrupt(PORT_SWITCH, pin);
    }
}


/**
 * Disables interrupt requests, i.e. masks them out at
 * the GPIO controller, triggered by the selected switch.
 * Pending interrupts for the pin are cleared.
 *
 * Nothing is done if 'sw' is invalid, i.e. different
 * from 1 or 2
 *
 * @param sw - desired switch (either 1 or 2)
 */
void switch_disableSwInt(uint8_t sw)
{
    const uint8_t pin =__sw2pin(sw);

    if ( PIN_UNKNOWN != pin )
    {
        gpio_maskInterrupt(PORT_SWITCH, pin);
        gpio_clearInterrupt(PORT_SWITCH, pin);
    }
}


/**
 * Registers a function that handles interrupt requests
 * triggered by the selected switch.
 *
 * Nothing is done if 'sw' is invalid, i.e. different from 1 or 2
 *
 * @param sw - desired switch (either 1 or 2)
 * @param isr - address of the interrupt handling routine
 */
void switch_registerIntrHandler(uint8_t sw, GpioPortIntHandler_t isr)
{
    const uint8_t pin =__sw2pin(sw);

    if ( PIN_UNKNOWN != pin )
    {
        gpio_registerIntHandler(PORT_SWITCH, pin, isr);
    }
}


/**
 * Unregisters the interrupt servicing routine for the specified
 * switch (actually it is replaced by an internal dummy function
 * that does not do anything).
 *
 * Nothing is done if 'sw' is invalid, i.e. different from 1 or 2
 *
 * @param sw - desired switch (either 1 or 2)
 */
void switch_unregisterIntrHandler(uint8_t sw)
{
    const uint8_t pin =__sw2pin(sw);

    if ( PIN_UNKNOWN != pin )
    {
        gpio_unregisterIntHandler(PORT_SWITCH, pin);
    }
}


/**
 * Clears the interrupt flag for the selected switch.
 *
 * Nothing is done if 'sw' is invalid, i.e. different from 1 or 2
 *
 * @param sw - desired switch (either 1 or 2)
 */
void switch_clearIntr(uint8_t sw)
{
    const uint8_t pin = __sw2pin(sw);

    if ( PIN_UNKNOWN != pin )
    {
        gpio_clearInterrupt(PORT_SWITCH, pin);
    }
}
