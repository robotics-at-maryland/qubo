/*
 * Jeremy Weed
 * jweed262@umd.edu
 * R@M 2017
 */

#include "lib/include/usb_serial.h"

void USB_serial_init(){

}

void USB_serial_configure(){
	/*
	ROM_SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOB);
	ROM_SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOG);
	ROM_GPIOPinConfigure(GPIO_PG4_USB0EPEN);
	ROM_GPIOPinTypeUSBDigital(GPIO_PORTG_BASE, GPIO_PIN_4);
	ROM_SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOL);
	ROM_GPIOPinTypeUSBAnalog(GPIO_PORTL_BASE, GPIO_PIN_6 | GPIO_PIN_7);
	ROM_GPIOPinTypeUSBAnalog(GPIO_PORTB_BASE, GPIO_PIN_0 | GPIO_PIN_1);

	// copied from the example I got most of this code from.  Unsure if its needed
	if(CLASS_IS_TM4C123 && REVISION_IS_A1){
		HWREG(GPIO_PORTB_BASE + GPIO_O_PDR) |= GPIO_PIN_1;
	}

	ROM_SysCtlPeripheralEnable(USB_UART_PERIPH);
	ROM_UARTConfigSetExpClk(USB_UART_BASE, ROM_SysCtlClockGet(), DEFAULT_BIT_RATE, DEFAULT_UART_CONFIG);
	ROM_UARTFIFOLevelSet(USB_UART_BASE, UART_FIFO_TX4_8, UART_FIFO_RX4_8);

	USBBufferInit(&USBTxBuffer);
	USBBufferInit(&USBRxBuffer);

	USBStackModeSet(0, eUSBModeDevice, 0);

	USBDCDCInit(0, &CDCDevice);
*/
}

uint32_t RxHandler(void *CBData, uint32_t event, uint32_t msgValue, void *msgData){
	switch( event ) {
		case USB_EVENT_RX_AVAILABLE: {
			//packet has been recieved
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
	}
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
			SetLineCoding(msgData);
			break;
		}
		case USBD_CDC_EVENT_SET_CONTROL_LINE_STATE: {
			SetControlLineState((uint16_t) msgValue);
			break;
		}
		case USBD_CDC_EVENT_SEND_BREAK: {
			SendBreak(true);
			break;
		}
		case USBD_CDC_EVENT_CLEAR_BREAK: {
			SendBreak(false);
			break;
		}
		case USB_EVENT_SUSPEND:
		case USB_EVENT_RESUME:
			//do nothing
			break;

	}
	return 0;
}
