/* Ross Baehr
   R@M 2017
   ross.baehr@gmail.com
*/

//QSCU
#include "include/configure.h"
#include "include/task_constants.h"
#include "tasks/include/read_uart0.h"
#include "tasks/include/read_uart1.h"
#include "tasks/include/example_blink.h"
#include "tasks/include/example_uart.h"
#include "tasks/include/i2c_test.h"

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
#include <driverlib/fpu.h>

// If debug defined, can use this to print to UART
#ifdef DEBUG
#include <utils/uartstdio.h>
#endif

// Globals
#include "include/i2c0_mutex.h"
#include "include/i2c1_mutex.h"
#include "include/i2c2_mutex.h"
#include "include/i2c3_mutex.h"

#include "include/i2c0_globals.h"
#include "include/i2c1_globals.h"
#include "include/i2c2_globals.h"
#include "include/i2c3_globals.h"

#include "include/uart0_mutex.h"
#include "include/uart1_mutex.h"
#include "include/rgb_mutex.h"
#include "include/read_uart0_queue.h"
#include "include/read_uart1_queue.h"

volatile uint32_t *i2c0_address;
volatile uint8_t **i2c0_buffer;
volatile uint32_t *i2c0_count;
volatile uint16_t *i2c0_int_state;

volatile uint32_t *i2c1_address;
volatile uint8_t **i2c1_buffer;
volatile uint32_t *i2c1_count;
volatile uint16_t *i2c1_int_state;

volatile uint32_t *i2c2_address;
volatile uint8_t **i2c2_buffer;
volatile uint32_t *i2c2_count;
volatile uint16_t *i2c2_int_state;

volatile uint32_t *i2c3_address;
volatile uint8_t **i2c3_buffer;
volatile uint32_t *i2c3_count;
volatile uint16_t *i2c3_int_state;

SemaphoreHandle_t i2c0_mutex;
SemaphoreHandle_t i2c1_mutex;
SemaphoreHandle_t i2c2_mutex;
SemaphoreHandle_t i2c3_mutex;

SemaphoreHandle_t uart0_mutex;
SemaphoreHandle_t uart1_mutex;
SemaphoreHandle_t rgb_mutex;

volatile QueueHandle_t read_uart0_queue;
volatile QueueHandle_t read_uart1_queue;


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

  // Enable floating point operations
  ROM_FPULazyStackingEnable();
  ROM_FPUEnable();

  // Set the clocking to run at 50 MHz from the PLL
  ROM_SysCtlClockSet(SYSCTL_SYSDIV_4 | SYSCTL_USE_PLL | SYSCTL_XTAL_16MHZ |
                     SYSCTL_OSC_MAIN);


  configureUART();
  configureGPIO();
  configureI2C();

  // Master enable interrupts
  ROM_IntMasterEnable();


  // -----------------------------------------------------------------------
  // Allocate FreeRTOS data structures for tasks, these are automatically made in heap
  // -----------------------------------------------------------------------

  i2c0_address = (uint32_t *)pvPortMalloc(sizeof(uint32_t *));
  // Double ptr, ptr to a pointer of buffer
  i2c0_buffer = (uint8_t **)pvPortMalloc(sizeof(uint8_t *));
  i2c0_count = (uint32_t *)pvPortMalloc(sizeof(uint32_t *));
  // This is the only variable that isn't existing in the calling task
  // Belongs to the interrupt, so we're not allocating a pointer but the value itself
  i2c0_int_state = (uint16_t *)pvPortMalloc(sizeof(uint16_t));

  i2c1_address = (uint32_t *)pvPortMalloc(sizeof(uint32_t *));
  // Double ptr, ptr to a pointer of buffer
  i2c1_buffer = (uint8_t **)pvPortMalloc(sizeof(uint8_t *));
  i2c1_count = (uint32_t *)pvPortMalloc(sizeof(uint32_t *));
  // This is the only variable that isn't existing in the calling task
  // Belongs to the interrupt, so we're not allocating a pointer but the value itself
  i2c1_int_state = (uint16_t *)pvPortMalloc(sizeof(uint16_t));

  i2c2_address = (uint32_t *)pvPortMalloc(sizeof(uint32_t *));
  // Double ptr, ptr to a pointer of buffer
  i2c2_buffer = (uint8_t **)pvPortMalloc(sizeof(uint8_t *));
  i2c2_count = (uint32_t *)pvPortMalloc(sizeof(uint32_t *));
  // This is the only variable that isn't existing in the calling task
  // Belongs to the interrupt, so we're not allocating a pointer but the value itself
  i2c2_int_state = (uint16_t *)pvPortMalloc(sizeof(uint16_t));

  i2c3_address = (uint32_t *)pvPortMalloc(sizeof(uint32_t *));
  // Double ptr, ptr to a pointer of buffer
  i2c3_buffer = (uint8_t **)pvPortMalloc(sizeof(uint8_t *));
  i2c3_count = (uint32_t *)pvPortMalloc(sizeof(uint32_t *));
  // This is the only variable that isn't existing in the calling task
  // Belongs to the interrupt, so we're not allocating a pointer but the value itself
  i2c3_int_state = (uint16_t *)pvPortMalloc(sizeof(uint16_t));

  i2c0_mutex = xSemaphoreCreateMutex();
  i2c1_mutex = xSemaphoreCreateMutex();
  i2c2_mutex = xSemaphoreCreateMutex();
  i2c3_mutex = xSemaphoreCreateMutex();

  uart0_mutex = xSemaphoreCreateMutex();
  uart1_mutex = xSemaphoreCreateMutex();
  rgb_mutex = xSemaphoreCreateMutex();

  read_uart0_queue = xQueueCreate(READ_UART0_Q_SIZE, sizeof(uint8_t));
  read_uart1_queue = xQueueCreate(READ_UART1_Q_SIZE, sizeof(uint8_t));

  #ifdef DEBUG
  UARTprintf("Datastructures allocated\n");
  #endif

  // -----------------------------------------------------------------------
  // Start FreeRTOS tasks
  // -----------------------------------------------------------------------

  if ( i2c_test_init() ) {
    while(1){}
  }

  /*
  if ( example_blink_init() ) {
    while(1){}
  }
  */

  /*
  if ( read_uart0_init() ) {
    while(1){}
  }
  */

  /*
  if ( example_uart_init() ) {
    while(1){}
  }
  */


  #ifdef DEBUG
  UARTprintf("\nTask's added, starting scheduler\n");
  #endif
  vTaskStartScheduler();

  while(1){}
}
