/* Ross Baehr
   R@M 2017
   ross.baehr@gmail.com
*/

//QSCU
#include "include/configure.h"
#include "include/task_constants.h"
#include "tasks/include/read_uart0.h"
#include "tasks/include/example_blink.h"
#include "tasks/include/example_uart.h"
#include "tasks/include/blink_red.h"
#include "tasks/include/blink_blue.h"

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
#include <inc/hw_nvic.h>
#include <driverlib/gpio.h>
#include <driverlib/i2c.h>
#include <driverlib/pin_map.h>
#include <driverlib/rom.h>
#include <driverlib/sysctl.h>
#include <driverlib/uart.h>

// If debug defined, can use this to print to UART
#ifdef DEBUG
#include <utils/uartstdio.h>
#endif

// Globals
#include "include/i2c0_mutex.h"
#include "include/uart0_mutex.h"
#include "include/uart1_mutex.h"
#include "include/rgb_mutex.h"
#include "include/read_uart0_queue.h"
#include "include/read_uart1_queue.h"

SemaphoreHandle_t i2c0_mutex;
SemaphoreHandle_t uart0_mutex;
SemaphoreHandle_t uart1_mutex;
SemaphoreHandle_t rgb_mutex;

QueueHandle_t read_uart0_queue;
QueueHandle_t read_uart1_queue;


#ifdef DEBUG
void __error__(char *pcFilename, uint32_t ui32Line)
{
  for (;;) {
    blink_rgb(RED_LED, 1);
    UARTprintf("ERROR");
  }
}
#endif

void vApplicationStackOverflowHook( TaskHandle_t pxTask, signed char *pcTaskName ) {
  for (;;) {}
}

// Called when a tick interrupt happens
// Can be used to confirm tick interrupt happening
void vApplicationTickHook(void) {
	#ifdef DEBUG
  //UARTprintf("\nTick interrupt\n");
	#endif
}

int main() {

  // Set the clocking to run at 50 MHz from the PLL
  ROM_SysCtlClockSet(SYSCTL_SYSDIV_4 | SYSCTL_USE_PLL | SYSCTL_XTAL_16MHZ |
                     SYSCTL_OSC_MAIN);


  configureUART();
  configureGPIO();
  configureI2C();

  // Master enable interrupts
  ROM_IntMasterEnable();

  #ifdef DEBUG
  UARTprintf("Active Interrupts: %x\n", HWREG(NVIC_ACTIVE0));
  UARTprintf("Active Interrupts: %x\n", HWREG(NVIC_ACTIVE1));
  UARTprintf("Active Interrupts: %x\n", HWREG(NVIC_ACTIVE2));
  UARTprintf("Active Interrupts: %x\n", HWREG(NVIC_ACTIVE3));
  UARTprintf("Active Interrupts: %x\n", HWREG(NVIC_ACTIVE4));
  #endif


  // -----------------------------------------------------------------------
  // Allocate FreeRTOS data structures for tasks, these are automatically made in heap
  // -----------------------------------------------------------------------

  uart0_mutex = xSemaphoreCreateMutex();
  uart1_mutex = xSemaphoreCreateMutex();
  i2c0_mutex = xSemaphoreCreateMutex();
  rgb_mutex = xSemaphoreCreateMutex();

  read_uart0_queue = xQueueCreate(READ_UART0_Q_SIZE, sizeof(uint8_t));
  read_uart1_queue = xQueueCreate(READ_UART1_Q_SIZE, sizeof(uint8_t));

  #ifdef DEBUG
  UARTprintf("Datastructures allocated\n");
  #endif

  // -----------------------------------------------------------------------
  // Start FreeRTOS tasks
  // -----------------------------------------------------------------------

  /*
  if ( example_blink_init() ) {
    while(1){}
  }
  */

  /*
  if ( read_uart_init() ) {
    while(1){}
  }
  */

  if ( example_uart_init() ) {
    while(1){}
  }


  #ifdef DEBUG
  UARTprintf("\nTask's added, starting scheduler\n");
  #endif
  vTaskStartScheduler();

  while(1){}
}
