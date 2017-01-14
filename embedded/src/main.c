//QSCU
#include "include/read_uart.h"
#include "include/write_uart.h"
#include "include/task_constants.h"

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
#include <driverlib/gpio.h>
#include <driverlib/i2c.h>
#include <driverlib/pin_map.h>
#include <driverlib/rom.h>
#include <driverlib/sysctl.h>
#include <driverlib/uart.h>
#include <utils/uartstdio.h>

#ifdef DEBUG
void __error__(char *pcFilename, uint32_t ui32Line)
{
}

#endif


void configureUART(void)
{
  //
  // Enable the GPIO Peripheral used by the UART.
  //
  ROM_SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOA);

  //
  // Enable UART0
  //
  ROM_SysCtlPeripheralEnable(SYSCTL_PERIPH_UART0);

  //
  // Configure GPIO Pins for UART mode.
  //
  ROM_GPIOPinConfigure(GPIO_PA0_U0RX);
  ROM_GPIOPinConfigure(GPIO_PA1_U0TX);
  ROM_GPIOPinTypeUART(GPIO_PORTA_BASE, GPIO_PIN_0 | GPIO_PIN_1);

  //
  // Use the internal 16MHz oscillator as the UART clock source.
  //
  UARTClockSourceSet(UART0_BASE, UART_CLOCK_PIOSC);

  //
  // Initialize the UART for console I/O.
  //
  UARTStdioConfig(0, 115200, 16000000);
}

void configureGPIO(void) {
  //
  // Enable the GPIO port that is used for the on-board LED.
  //
  ROM_SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOF);

  //
  // Enable the GPIO pins for the LED (PF2).
  //
  ROM_GPIOPinTypeGPIOOutput(GPIO_PORTF_BASE, GPIO_PIN_2);

}

void configureI2C(void) {
  // Enable i2c periph
  SysCtlPeripheralEnable(SYSCTL_PERIPH_I2C0);

  // I2C0 With PortB[3:2]
  SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOB);

  // Configure pins for scl and sda
  GPIOPinConfigure(GPIO_PB2_I2C0SCL);
  GPIOPinConfigure(GPIO_PB3_I2C0SDA);

  // Select the I2C function for these pins.  This function will also
  // configure the GPIO pins pins for I2C operation, setting them to
  // open-drain operation with weak pull-ups.  Consult the data sheet
  // to see which functions are allocated per pin.
  GPIOPinTypeI2CSCL(GPIO_PORTB_BASE, GPIO_PIN_2);
  GPIOPinTypeI2C(GPIO_PORTB_BASE, GPIO_PIN_3);

  // Enable and initialize the I2C0 master module.  Use the system clock for
  // the I2C0 module.  The last parameter sets the I2C data transfer rate.
  // If false the data rate is set to 100kbps and if true the data rate will
  // be set to 400kbps.
  I2CMasterInitExpClk(I2C0_BASE, SysCtlClockGet(), false);

}


void vApplicationStackOverflowHook( TaskHandle_t pxTask, signed char *pcTaskName ) {
  for (;;) {}
}

int main() {
  ROM_SysCtlClockSet(SYSCTL_SYSDIV_4 | SYSCTL_USE_PLL | SYSCTL_XTAL_16MHZ |
                     SYSCTL_OSC_MAIN);

  // Master enable interrupts
  IntMasterEnable();

  configureUART();
  configureGPIO();

  // -----------------------------------------------------------------------
  // Allocate FreeRTOS data structures for tasks, this may be changed to dynamic
  // -----------------------------------------------------------------------

  //  read_uart = xQueueCreate(READ_UART_Q_SIZE, sizeof(int32_t));
  //  write_uart = xQueueCreate(WRITE_UART_Q_SIZE, sizeof(int32_t));

  // -----------------------------------------------------------------------
  // Start FreeRTOS tasks
  // -----------------------------------------------------------------------
  if ( xTaskCreate(read_uart_task, (const portCHAR *)"Read_UART", READ_UART_STACKSIZE,
                   NULL, READ_UART_PRIORITY, NULL) != pdTRUE) {
    // ERROR
  }

  vTaskStartScheduler();
  for (;;) {}
}
