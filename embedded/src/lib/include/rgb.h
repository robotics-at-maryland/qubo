/* Ross Baehr
   R@M 2017
   ross.baehr@gmail.com
*/

#ifndef _RGB_H_
#define _RGB_H_

// Tiva
#include <stdbool.h>
#include <stdint.h>
#include <inc/hw_memmap.h>
#include <inc/hw_types.h>
#include <driverlib/gpio.h>
#include <driverlib/pin_map.h>
#include <driverlib/rom.h>
#include <driverlib/sysctl.h>
#include <driverlib/uart.h>
#include <utils/uartstdio.h>

#include "include/rgb_mutex.h"

#include <FreeRTOS.h>
#include <semphr.h>
#include <task.h>

#define RED_LED   GPIO_PIN_1
#define BLUE_LED  GPIO_PIN_2
#define GREEN_LED GPIO_PIN_3

#define BLINK_RATE 250

volatile SemaphoreHandle_t blink_mutex;

// Blink the RGB led color c and n times
void blink_rgb(uint8_t color, uint8_t n);

void rgb_on(uint8_t color);

void rgb_off(uint8_t color);

// Blink the RGB led color c and n times from an isr
void blink_rgb_from_isr(uint8_t color, uint8_t n);

void rgb_on_from_isr(uint8_t color);

void rgb_off_from_isr(uint8_t color);

#endif
