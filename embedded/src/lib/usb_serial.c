/*
 * Jeremy Weed
 * jweed262@umd.edu
 * R@M 2017
 */

#include "lib/include/usb_serial.h"

// regex for easy editing: (\w) '$1', 0,
const uint8_t lang_descr[] = { 4, USB_DTYPE_STRING, USBShort(USB_LANG_EN_US) };

const uint8_t manufac_str[] = { 2 + 3 * 2, USB_DTYPE_STRING, 'R', 0, '@', 0, 'M', 0 };

const uint8_t product_str[] = { 2 + 4 * 2, USB_DTYPE_STRING, 'T', 0, 'i', 0, 'v', 0, 'a', 0};

const uint8_t serial_num[] = { 2 + 4 * 2, USB_DTYPE_STRING, '2', 0, '0', 0, '1', 0, '7', 0 };

const uint8_t contr_intrfc_str[] = { 2 + 4 * 2, USB_DTYPE_STRING, 'c', 0, 'o', 0, 'm', 0, 'p', 0};

const uint8_t config_desc_str[] = { 2 + 4 * 2, USB_DTYPE_STRING, 'D', 0, 'E', 0, 'S', 0, 'C', 0};

const uint8_t * const str_desc[] = {
	lang_descr,
	manufac_str,
	product_str,
	serial_num,
	contr_intrfc_str,
	config_desc_str
};

#define NUM_DESC ( sizeof(str_desc) / sizeof(uint8_t *) )

const tUSBDCDCDevice CDCDevice = {
// The Vendor ID you have been assigned by USB-IF.
USB_VID_TI_1CBE,
// The product ID you have assigned for this device.
USB_PID_SERIAL,
// The power consumption of your device in milliamps.
0,
// The value to be passed to the host in the USB configuration descriptor’s
// bmAttributes field.
USB_CONF_ATTR_SELF_PWR,
// A pointer to your control callback event handler.
ControlHandler,
// A value that you want passed to the control callback alongside every
// event.
(void *)&CDCDevice,
// A pointer to your receive callback event handler.
USBBufferEventCallback,
// A value that you want passed to the receive callback alongside every
// event.
(void *)&USBRxBuffer,
// A pointer to your transmit callback event handler.
USBBufferEventCallback,
// A value that you want passed to the transmit callback alongside every
// event.
(void *)&USBTxBuffer,
// A pointer to your string table.
str_desc,
// The number of entries in your string table.
NUM_DESC
};



uint8_t rxBuffer[QUBOBUS_MAX_PAYLOAD_LENGTH];

const tUSBBuffer USBRxBuffer = {
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

const tUSBBuffer USBTxBuffer = {
    true,                           // This is a transmit buffer.
    RxHandler,                      // pfnCallback
    (void *)&CDCDevice,          	// Callback data is our device pointer.
    USBDCDCPacketWrite,             // pfnTransfer
    USBDCDCTxPacketAvailable,       // pfnAvailable
    (void *)&CDCDevice,          	// pvHandle
    txBuffer,               		// pi8Buffer
    QUBOBUS_MAX_PAYLOAD_LENGTH,     // ui32BufferSize
};

void * USB_CDC = NULL;

uint8_t USB_serial_init(void *device){
	if ( xTaskCreate(usb_task, (const portCHAR *)"USB task", 128, NULL,
                     tskIDLE_PRIORITY + 1, NULL) != pdTRUE) {
        return true;
    }
	#ifdef DEBUG
	UARTprintf("serial init\n");
	#endif
	USBBufferInit(&USBTxBuffer);
	USBBufferInit(&USBRxBuffer);

	USBStackModeSet(0, eUSBModeForceDevice, 0);
	USB_CDC = USBDCDCInit(0, &CDCDevice);
	if( NULL != USB_CDC ){
		#ifdef DEBUG
		UARTprintf("init completed\n");
		#endif
	} else {
		#ifdef DEBUG
		UARTprintf("init failed\n");
		#endif
	}
 
	return 0;
}

static void usb_task(void * params){
	uint8_t val = '0';
	for(;;){
		USBBufferWrite(&USBTxBuffer, &val, sizeof(val));
		blink_rgb(GREEN_LED | RED_LED, 1);
	}
}

void USB_serial_configure(){

	ROM_SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOD);
	ROM_GPIOPinTypeUSBAnalog(GPIO_PORTD_BASE, GPIO_PIN_5 | GPIO_PIN_4);

}

// Copied from the usb example.  Returns the configuration of UART0,
// because I couldn't thing of anything better
static void GetLineCoding(tLineCoding *psLineCoding) {
    uint32_t ui32Config;
    uint32_t ui32Rate;
    // Get the current line coding set in the UART.
    ROM_UARTConfigGetExpClk(UART0_BASE, ROM_SysCtlClockGet(), &ui32Rate,
                            &ui32Config);
    psLineCoding->ui32Rate = ui32Rate;
    // Translate the configuration word length field into the format expected
    // by the host.
    switch(ui32Config & UART_CONFIG_WLEN_MASK){
        case UART_CONFIG_WLEN_8: {
            psLineCoding->ui8Databits = 8;
            break;
        }

        case UART_CONFIG_WLEN_7: {
            psLineCoding->ui8Databits = 7;
            break;
        }

        case UART_CONFIG_WLEN_6: {
            psLineCoding->ui8Databits = 6;
            break;
        }

        case UART_CONFIG_WLEN_5: {
            psLineCoding->ui8Databits = 5;
            break;
        }
    }
    // Translate the configuration parity field into the format expected
    // by the host.
    switch(ui32Config & UART_CONFIG_PAR_MASK) {
        case UART_CONFIG_PAR_NONE: {
            psLineCoding->ui8Parity = USB_CDC_PARITY_NONE;
            break;
        }

        case UART_CONFIG_PAR_ODD: {
            psLineCoding->ui8Parity = USB_CDC_PARITY_ODD;
            break;
        }

        case UART_CONFIG_PAR_EVEN: {
            psLineCoding->ui8Parity = USB_CDC_PARITY_EVEN;
            break;
        }

        case UART_CONFIG_PAR_ONE: {
            psLineCoding->ui8Parity = USB_CDC_PARITY_MARK;
            break;
        }

        case UART_CONFIG_PAR_ZERO: {
            psLineCoding->ui8Parity = USB_CDC_PARITY_SPACE;
            break;
        }
    }
    // Translate the configuration stop bits field into the format expected
    // by the host.
    switch(ui32Config & UART_CONFIG_STOP_MASK) {
        case UART_CONFIG_STOP_ONE: {
            psLineCoding->ui8Stop = USB_CDC_STOP_BITS_1;
            break;
        }

        case UART_CONFIG_STOP_TWO: {
            psLineCoding->ui8Stop = USB_CDC_STOP_BITS_2;
            break;
        }
    }
}

uint32_t RxHandler(void *CBData, uint32_t event, uint32_t msgValue, void *msgData){
	switch( event ) {
		case USB_EVENT_RX_AVAILABLE: {
			//uint32_t p_length = USBDCDCRxPacketAvailable(USB_CDC);
			#ifdef DEBUG
			UARTprintf("packet p_length");
			#endif
			uint8_t data;
			USBBufferRead(&USBRxBuffer, &data, sizeof(data));
			#ifdef DEBUG
			UARTprintf("data: %i\n", data);
			#endif

			break;
		}
		case USB_EVENT_DATA_REMAINING: {
			// return how much data we need to process
			break;
		}
		case USB_EVENT_REQUEST_BUFFER: {
			// no buffer available, so return 0
			return 0;
		}
		default: {
			#ifdef DEBUG
			UARTprintf("USB RX: we shouldn't be here\n");
			#endif
		}
	}
	return 0;
}

uint32_t TxHandler(void *CBData, uint32_t event, uint32_t msgValue, void *msgData){
	switch( event ) {
		case USB_EVENT_TX_COMPLETE:
			#ifdef DEBUG
			UARTprintf("finished USB write\n");
			#endif
			break;
		default:
			#ifdef DEBUG
			UARTprintf("USB TX: we shouldn't be here\n");
			#endif
			break;
	}
	return 0;
}

uint32_t ControlHandler(void *CBData, uint32_t event, uint32_t msgValue, void *msgData){
	switch ( event ){
		case USB_EVENT_CONNECTED: {
			// we have connected
			USBBufferFlush(&USBTxBuffer);
			USBBufferFlush(&USBRxBuffer);

			#ifdef DEBUG
			UARTprintf("USB Connected\n");
			#endif
			break;
		}
		case USB_EVENT_DISCONNECTED: {
			// connection lost
			#ifdef DEBUG
			UARTprintf("USB Disconnected\n");
			#endif
			break;
		}
		case USBD_CDC_EVENT_GET_LINE_CODING: {
			GetLineCoding(msgData);
			break;
		}
		case USBD_CDC_EVENT_SET_LINE_CODING: {
			break;
		}
		case USBD_CDC_EVENT_SET_CONTROL_LINE_STATE: {
			break;
		}
		case USBD_CDC_EVENT_SEND_BREAK: {
			break;
		}
		case USBD_CDC_EVENT_CLEAR_BREAK: {
			break;
		}
		case USB_EVENT_SUSPEND:
		case USB_EVENT_RESUME:
			//do nothing
			break;
		default:{
			#ifdef DEBUG
			UARTprintf("Something happened on the USB, but we didn't catch it\n");
			#endif
		}
	}
	return 0;
}
