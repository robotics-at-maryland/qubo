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
 * Declaration of public functions that handle
 * all 6 General-Purpose Input/Output (GPIO) ports.
 *
 * @author Jernej Kovacic
 */


#ifndef _GPIO_H_
#define _GPIO_H_

#include <stdint.h>


/**
 * An enumeration with supported GPIO
 * pad drive strengths for digital communication.
 */
typedef enum _gpio_drive_t
{
    DR_2_MA,      /** 2 mA */
    DR_4_MA,      /** 4 mA */
    DR_8_MA       /** 8 mA */
} gpio_drive_t;


/**
 * Convenience definitions for mapping between ports's
 * names ('A' to 'F') and their numeric values.
 */
#define GPIO_PORTA       ( 0 )
#define GPIO_PORTB       ( 1 )
#define GPIO_PORTC       ( 2 )
#define GPIO_PORTD       ( 3 )
#define GPIO_PORTE       ( 4 )
#define GPIO_PORTF       ( 5 )


/**
 * An enumeration with all supported modes for
 * detection of GPIO interrupt events.
 */
typedef enum _gpioIntType
{
    GPIOINT_LEVEL_HIGH,         /** Interrupt triggered on low level (0)           */
    GPIOINT_LEVEL_LOW,          /** Interrupt triggered on high level (1)          */
    GPIOINT_EDGE_FALLING,       /** Interrupt triggered on falling edge (1->0)     */
    GPIOINT_EDGE_RISING,        /** Interrupt triggered on rising edge (0->1)      */
    GPIOINT_EDGE_ANY            /** Interrupt triggered on any edge (0->1 or 1->0) */
} gpioIntType;


/**
 * Required prototype for GPIO triggered interrupt requests.
 */
typedef void (*GpioPortIntHandler_t)(void);



void gpio_enablePort(uint8_t port);

void gpio_disablePort(uint8_t port);

void gpio_setPinAsInput(uint8_t port, uint8_t pin);

void gpio_setPinAsOutput(uint8_t port, uint8_t pin);

void gpio_setAltFunction(uint8_t port, uint8_t pin, uint8_t pctl);

void gpio_enableDigital(uint8_t port, uint8_t pin);

void gpio_disableDigital(uint8_t port, uint8_t pin);

void gpio_enableAnalog(uint8_t port, uint8_t pin);

void gpio_disableAnalog(uint8_t port, uint8_t pin);

void gpio_enablePullUp(uint8_t port, uint8_t pin);

void gpio_disablePullUp(uint8_t port, uint8_t pin);

void gpio_enablePullDown(uint8_t port, uint8_t pin);

void gpio_disablePullDown(uint8_t port, uint8_t pin);

void gpio_enableOpenDrain(uint8_t port, uint8_t pin);

void gpio_disableOpenDrain(uint8_t port, uint8_t pin);

void gpio_setDrive(uint8_t port, uint8_t pin, gpio_drive_t dr);

void gpio_setPin(uint8_t port, uint8_t pin);

void gpio_clearPin(uint8_t port, uint8_t pin);

void gpio_setPins(uint8_t port, uint8_t pinmask);

void gpio_clearPins(uint8_t port, uint8_t pinmask);

uint8_t gpio_readPin(uint8_t port, uint8_t pin);

uint8_t gpio_readPins(uint8_t port, uint8_t pinmask);

void gpio_enableInterruptPort(uint8_t port);

void gpio_disableInterruptPort(uint8_t port);

void gpio_setIntrPriority(uint8_t port, uint8_t pri);

void gpio_unmaskInterrupt(uint8_t port, uint8_t pin);

void gpio_maskInterrupt(uint8_t port, uint8_t pin);

void gpio_clearInterrupt(uint8_t port, uint8_t pin);

void gpio_configInterrupt(uint8_t port, uint8_t pin, gpioIntType mode);

void gpio_registerIntHandler(uint8_t port, uint8_t pin, GpioPortIntHandler_t isr);

void gpio_unregisterIntHandler(uint8_t port, uint8_t pin);

void gpio_configGpioPin(uint8_t port, uint8_t pin, uint8_t output);

#endif /* _GPIO_H_ */
