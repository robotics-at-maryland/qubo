/*
 * Jeremy Weed
 * jweed262@umd.edu
 * R@M 2017
 */

#ifndef USB_SERIAL_H
#define USB_SERIAL_H

// FreeRTOS
#include <FreeRTOS.h>
#include <queue.h>
#include <task.h>
#include <semphr.h>

// Tiva
#include <stdbool.h>
#include <stdint.h>
#include <inc/hw_ints.h>
#include <inc/hw_memmap.h>
#include <inc/hw_types.h>
#include <inc/hw_gpio.h>
#include <inc/hw_uart.h>
#include <inc/hw_sysctl.h>
#include <driverlib/gpio.h>
#include <driverlib/pin_map.h>
#include <driverlib/interrupt.h>
#include <driverlib/rom.h>
#include <driverlib/sysctl.h>
#include <driverlib/uart.h>
#include <driverlib/usb.h>
#include <usblib/usblib.h>
#include <usblib/usbcdc.h>
#include <usblib/usb-ids.h>
#include <usblib/device/usbdevice.h>
#include <usblib/device/usbdcdc.h>
#include "utils/ustdlib.h"

// qubobus
#include <sys/types.h>
#include "qubobus.h"
#include "io.h"

#include "lib/include/rgb.h"
#ifdef DEBUG
#include <utils/uartstdio.h>
#endif

extern const tUSBBuffer USBTxBuffer;
extern const tUSBBuffer USBRxBuffer;

extern void* USB_CDC;

extern const tUSBDCDCDevice CDCDevice;

uint32_t RxHandler(void *CBData, uint32_t event, uint32_t msgValue, void *msgData);
uint32_t TxHandler(void *CBData, uint32_t event, uint32_t msgValue, void *msgData);
uint32_t ControlHandler(void *CBData, uint32_t event, uint32_t msgValue, void *msgData);

uint8_t USB_serial_init();
void USB_serial_configure();
static void usb_task(void *params);

#endif
