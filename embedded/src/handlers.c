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
 * Implementation of exception and interrupt handler functions.
 *
 * @author Jernej Kovacic
 */


#include <stdint.h>

#include "gpio.h"


/* forward declaration of an "unofficial" function from gpio.c */
extern void _gpio_intHandler(uint8_t port);

/* forward declaration of an "unofficial" function from receive.c */
extern void _recv_intHandler(uint8_t uart);


/*
 * Handler for interrupts, triggered by a pin
 * from the GPIO port A.
 */
__attribute__ ((interrupt))
void GpioAIntHandler(void)
{
    _gpio_intHandler(GPIO_PORTA);
}


/*
 * Handler for interrupts, triggered by a pin
 * from the GPIO port B.
 */
__attribute__ ((interrupt))
void GpioBIntHandler(void)
{
    _gpio_intHandler(GPIO_PORTB);
}


/*
 * Handler for interrupts, triggered by a pin
 * from the GPIO port C.
 */
__attribute__ ((interrupt))
void GpioCIntHandler(void)
{
    _gpio_intHandler(GPIO_PORTC);
}


/*
 * Handler for interrupts, triggered by a pin
 * from the GPIO port D.
 */
__attribute__ ((interrupt))
void GpioDIntHandler(void)
{
    _gpio_intHandler(GPIO_PORTD);
}


/*
 * Handler for interrupts, triggered by a pin
 * from the GPIO port E.
 */
__attribute__ ((interrupt))
void GpioEIntHandler(void)
{
    _gpio_intHandler(GPIO_PORTE);
}


/*
 * Handler for interrupts, triggered by a pin
 * from the GPIO port F.
 */
__attribute__ ((interrupt))
void GpioFIntHandler(void)
{
    _gpio_intHandler(GPIO_PORTF);
}



/*
 * Handler for interrupts, triggered by UART0.
 */
__attribute__ ((interrupt))
void Uart0IntHandler(void)
{
    _recv_intHandler(0);
}


/*
 * Handler for interrupts, triggered by UART1.
 */
__attribute__ ((interrupt))
void Uart1IntHandler(void)
{
    _recv_intHandler(1);
}


/*
 * Handler for interrupts, triggered by UART2.
 */
__attribute__ ((interrupt))
void Uart2IntHandler(void)
{
    _recv_intHandler(2);
}


/*
 * Handler for interrupts, triggered by UART3.
 */
__attribute__ ((interrupt))
void Uart3IntHandler(void)
{
    _recv_intHandler(3);
}


/*
 * Handler for interrupts, triggered by UART4.
 */
__attribute__ ((interrupt))
void Uart4IntHandler(void)
{
    _recv_intHandler(4);
}


/*
 * Handler for interrupts, triggered by UART5.
 */
__attribute__ ((interrupt))
void Uart5IntHandler(void)
{
    _recv_intHandler(5);
}


/*
 * Handler for interrupts, triggered by UART6.
 */
__attribute__ ((interrupt))
void Uart6IntHandler(void)
{
    _recv_intHandler(6);
}


/*
 * Handler for interrupts, triggered by UART7.
 */
__attribute__ ((interrupt))
void Uart7IntHandler(void)
{
    _recv_intHandler(7);
}
