//QSCU
#include "include/task_constants.h"
#include "tasks/include/read_uart.h"
#include "tasks/include/example_blink.h"
#include "tasks/include/example_uart.h"

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

// Globals
#include "include/i2c_mutex.h"
#include "include/uart_mutex.h"
#include "include/blink_mutex.h"
SemaphoreHandle_t i2c_mutex;
SemaphoreHandle_t uart_mutex;
SemaphoreHandle_t blink_mutex;


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
  ROM_UARTClockSourceSet(UART0_BASE, UART_CLOCK_PIOSC);

  //
  // Initialize the UART for console I/O.
  //
  UARTStdioConfig(0, 115200, 16000000);
}

void configureGPIO(void) {

  ROM_SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOF);
  while(!ROM_SysCtlPeripheralReady(SYSCTL_PERIPH_GPIOF))
    {
    }

  //
  // Configure the GPIO port for the LED operation.
  //
  ROM_GPIOPinTypeGPIOOutput(GPIO_PORTF_BASE, GPIO_PIN_1|GPIO_PIN_2|GPIO_PIN_3);

}

void vApplicationStackOverflowHook( TaskHandle_t pxTask, signed char *pcTaskName ) {
  for (;;) {}
}

int main() {

  // Set the clocking to run at 50 MHz from the PLL
  ROM_SysCtlClockSet(SYSCTL_SYSDIV_4 | SYSCTL_USE_PLL | SYSCTL_XTAL_16MHZ |
                     SYSCTL_OSC_MAIN);

  // Master enable interrupts
  ROM_IntMasterEnable();

   configureUART();
   configureGPIO();

  // Query i2c init
  //  initI2C();

   UARTprintf("\n\nTask running\n");



  // -----------------------------------------------------------------------
  // Allocate FreeRTOS data structures for tasks, this may be changed to dynamic
  // -----------------------------------------------------------------------

  uart_mutex = xSemaphoreCreateMutex();
  i2c_mutex = xSemaphoreCreateMutex();
  blink_mutex = xSemaphoreCreateMutex();

  // -----------------------------------------------------------------------
  // Start FreeRTOS tasks
  // -----------------------------------------------------------------------
  //  if ( xTaskCreate(read_uart_task, (const portCHAR *)"Read_UART", READ_UART_STACKSIZE,
  //                   NULL, READ_UART_PRIORITY, NULL) != pdTRUE) {
    // ERROR
  //  }


  if ( example_blink_init() ) {
    while(1){}
  }


  /*
  if ( read_uart_init() ) {
    while(1){}
  }
  */

  /*
  if ( example_uart_init() ) {
    while(1){}
  }
  */

  vTaskStartScheduler();

  while(1){}
}
