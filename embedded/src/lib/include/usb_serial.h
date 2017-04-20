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
#include <inc/hw_memmap.h>
#include <inc/hw_types.h>
#include <inc/hw_ints.h>
#include <inc/hw_sysctl.h>
#include <driverlib/gpio.h>
#include <driverlib/pin_map.h>
#include <driverlib/rom.h>
#include <driverlib/sysctl.h>
#include <driverlib/uart.h>
#include <driverlib/usb.h>
#include <usblib/usblib.h>
#include <usblib/usbcdc.h>
#include <usblib/usb-ids.h>
#include <usblib/device/usbdevice.h>
#include <usblib/device/usbdcdc.h>

#include <sys/types.h>
#include "qubobus.h"
#include "io.h"

#ifdef DEBUG
#include <utils/uartstdio.h>
#endif
// regex for easy editing: (\w) '$1', '0',

const uint8_t lang_descr[] = { 4, USB_DTYPE_STRING, USBShort(USB_LANG_EN_US) };

const uint8_t manufac_str[] = { 2 + 3 * 2, USB_DTYPE_STRING, 'R', '0', '@', 'M', '0' };

const uint8_t product_str[] = { 2 + 4 * 2, USB_DTYPE_STRING, 'T', '0', 'i', '0', 'v', 'a', '0'};

const uint8_t serial_num[] = { 2 + 4 * 2, USB_DTYPE_STRING, '2', '0', '0', '0', '1', '7', '0' };

const uint8_t contr_intrfc_str[] = { 2+ 4 * 2, USB_DTYPE_STRING, 'A', '0', 'C', '0', 'M', '0', '0', '0'};

const uint8_t config_desc_str[] = { 2 + 4 * 2, USB_DTYPE_STRING, 'D', '0','E', '0','S', '0','C', '0'};

const uint8_t * const str_desc[] = {
	lang_descr,
	manufac_str,
	product_str,
	serial_num,
	contr_intrfc_str,
	config_desc_str
};

#define NUM_DESC ( sizeof(str_desc) / sizeof(uint8_t *) )

extern const tUSBBuffer USBTxBuffer;
extern const tUSBBuffer USBRxBuffer;

uint32_t RxHandler(void *CBData, uint32_t event, uint32_t msgValue, void *msgData);
uint32_t TxHandler(void *CBData, uint32_t event, uint32_t msgValue, void *msgData);
uint32_t ControlHandler(void *CBData, uint32_t event, uint32_t msgValue, void *msgData);

const tUSBDCDCDevice CDCDevice = {
//
// The Vendor ID you have been assigned by USB-IF.
//
USB_VID_TI_1CBE,
//
// The product ID you have assigned for this device.
//
USB_PID_SERIAL,
//
// The power consumption of your device in milliamps.
//
0,
//
// The value to be passed to the host in the USB configuration descriptor’s
// bmAttributes field.
//
USB_CONF_ATTR_SELF_PWR,
//
// A pointer to your control callback event handler.
//
ControlHandler,
//
// A value that you want passed to the control callback alongside every
// event.
//
(void *)&CDCDevice,
//
// A pointer to your receive callback event handler.
//
USBBufferEventCallback,
//
// A value that you want passed to the receive callback alongside every
// event.
//
(void *)&USBRxBuffer,
//
// A pointer to your transmit callback event handler.
//
USBBufferEventCallback,
//
// A value that you want passed to the transmit callback alongside every
// event.
//
(void *)&USBTxBuffer,
//
// A pointer to your string table.
//
str_desc,
//
// The number of entries in your string table.
//
NUM_DESC
};



uint8_t rxBuffer[QUBOBUS_MAX_PAYLOAD_LENGTH];

const tUSBBuffer USBTxBuffer = {
	false,                          // This is a receive buffer.
	RxHandler,                      // pfnCallback
	(void *)&CDCDevice,             // Callback data is our device pointer.
	USBDCDCPacketRead,              // pfnTransfer
	USBDCDCRxPacketAvailable,       // pfnAvailable
	(void *)&CDCDevice,             // pvHandle
	rxBuffer,               		// pi8Buffer
	QUBOBUS_MAX_PAYLOAD_LENGTH,     // ui32BufferSize
};

uint8_t txBuffer[QUBOBUS_MAX_PAYLOAD_LENGTH];

const tUSBBuffer USBRxBuffer = {
    true,                           // This is a transmit buffer.
    TxHandler,                      // pfnCallback
    (void *)&CDCDevice,          	// Callback data is our device pointer.
    USBDCDCPacketWrite,             // pfnTransfer
    USBDCDCTxPacketAvailable,       // pfnAvailable
    (void *)&CDCDevice,          	// pvHandle
    txBuffer,               		// pi8Buffer
    QUBOBUS_MAX_PAYLOAD_LENGTH,     // ui32BufferSize
};
#endif
