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
#include "tasks/include/tiqu.h"
#include "tasks/include/thruster_task.h"
#include "lib/include/usb_serial.h"

// FreeRTOS
#include <FreeRTOS.h>
#include <queue.h>
#include <task.h>
#include <semphr.h>
#include <message_buffer.h>

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

#include "lib/include/uart_queue.h"
#include "interrupts/include/uart0_interrupt.h"

// Globals
#include "include/i2c0_mutex.h"
#include "include/i2c1_mutex.h"
#include "include/i2c2_mutex.h"
#include "include/i2c3_mutex.h"

#include "include/i2c0_globals.h"
#include "include/i2c1_globals.h"
#include "include/i2c2_globals.h"
#include "include/i2c3_globals.h"

#include "include/uart1_mutex.h"
#include "include/rgb_mutex.h"
#include "include/read_uart1_queue.h"

/* #include "include/task_handles.h" */
/* #include "include/task_queues.h" */
/* #include "tasks/include/qubobus_test.h" */


SemaphoreHandle_t i2c0_mutex;
SemaphoreHandle_t i2c1_mutex;
SemaphoreHandle_t i2c2_mutex;
SemaphoreHandle_t i2c3_mutex;

SemaphoreHandle_t uart1_mutex;
SemaphoreHandle_t rgb_mutex;

volatile uint32_t *i2c0_address;
volatile uint8_t **i2c0_read_buffer;
volatile uint8_t **i2c0_write_buffer;
volatile uint32_t *i2c0_read_count;
volatile uint32_t *i2c0_write_count;
volatile uint16_t *i2c0_int_state;

volatile uint32_t *i2c1_address;
volatile uint8_t **i2c1_read_buffer;
volatile uint8_t **i2c1_write_buffer;
volatile uint32_t *i2c1_read_count;
volatile uint32_t *i2c1_write_count;
volatile uint16_t *i2c1_int_state;

volatile uint32_t *i2c2_address;
volatile uint8_t **i2c2_read_buffer;
volatile uint8_t **i2c2_write_buffer;
volatile uint32_t *i2c2_read_count;
volatile uint32_t *i2c2_write_count;
volatile uint16_t *i2c2_int_state;

volatile uint32_t *i2c3_address;
volatile uint8_t **i2c3_read_buffer;
volatile uint8_t **i2c3_write_buffer;
volatile uint32_t *i2c3_read_count;
volatile uint32_t *i2c3_write_count;
volatile uint16_t *i2c3_int_state;


volatile struct UART_Queue uart0_queue;
volatile struct UART_Queue uart1_queue;

MessageBufferHandle_t thruster_message_buffer;

/* DECLARE_TASK_HANDLES; */
/* DECLARE_TASK_QUEUES; */

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
  for (;;) {
#ifdef DEBUG
    UARTprintf("\nTick interrupt\n");
#endif
  }
}

// Called when a tick interrupt happens
// Can be used to confirm tick interrupt happening
void vApplicationTickHook(void) {
#ifdef DEBUG
  UARTprintf("\nTick interrupt\n");
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
  /* USB_serial_configure(); */

  // Master enable interrupts
  ROM_IntMasterEnable();


  // -----------------------------------------------------------------------
  // Allocate FreeRTOS data structures for tasks, these are automatically made in heap
  // -----------------------------------------------------------------------

  i2c0_mutex  = xSemaphoreCreateMutex();
  i2c1_mutex  = xSemaphoreCreateMutex();
  i2c2_mutex  = xSemaphoreCreateMutex();
  i2c3_mutex  = xSemaphoreCreateMutex();
  uart1_mutex = xSemaphoreCreateMutex();
  rgb_mutex   = xSemaphoreCreateMutex();




  // Initialize the UART Queue for UART0.
  INIT_UART_QUEUE(uart0_queue, 256, 256, INT_UART0, UART0_BASE, pdMS_TO_TICKS(1000));
  INIT_UART_QUEUE(uart1_queue, 256, 256, INT_UART1, UART1_BASE, pdMS_TO_TICKS(1000));

  /* INIT_TASK_QUEUES(); */

  thruster_message_buffer = xMessageBufferCreate(8 * sizeof(struct Thruster_Set));

  i2c0_address      = pvPortMalloc(sizeof(uint32_t));
  i2c0_read_buffer  = pvPortMalloc(sizeof(uint8_t*));
  i2c0_write_buffer = pvPortMalloc(sizeof(uint8_t*));
  i2c0_read_count   = pvPortMalloc(sizeof(uint32_t));
  i2c0_write_count  = pvPortMalloc(sizeof(uint32_t));
  i2c0_int_state    = pvPortMalloc(sizeof(uint16_t));

  i2c1_address      = pvPortMalloc(sizeof(uint32_t));
  i2c1_read_buffer  = pvPortMalloc(sizeof(uint8_t*));
  i2c1_write_buffer = pvPortMalloc(sizeof(uint8_t*));
  i2c1_read_count   = pvPortMalloc(sizeof(uint32_t));
  i2c1_write_count  = pvPortMalloc(sizeof(uint32_t));
  i2c1_int_state    = pvPortMalloc(sizeof(uint16_t));

  i2c2_address      = pvPortMalloc(sizeof(uint32_t));
  i2c2_read_buffer  = pvPortMalloc(sizeof(uint8_t*));
  i2c2_write_buffer = pvPortMalloc(sizeof(uint8_t*));
  i2c2_read_count   = pvPortMalloc(sizeof(uint32_t));
  i2c2_write_count  = pvPortMalloc(sizeof(uint32_t));
  i2c2_int_state    = pvPortMalloc(sizeof(uint16_t));

  i2c3_address      = pvPortMalloc(sizeof(uint32_t));
  i2c3_read_buffer  = pvPortMalloc(sizeof(uint8_t*));
  i2c3_write_buffer = pvPortMalloc(sizeof(uint8_t*));
  i2c3_read_count   = pvPortMalloc(sizeof(uint32_t));
  i2c3_write_count  = pvPortMalloc(sizeof(uint32_t));
  i2c3_int_state    = pvPortMalloc(sizeof(uint16_t));

#ifdef DEBUG
  UARTprintf("Datastructures allocated\n");
#endif

  /* blink_rgb(BLUE_LED, 1); */
  // -----------------------------------------------------------------------
  // Start FreeRTOS tasks
  // -----------------------------------------------------------------------

  /* if ( i2c_test_init() ) { */
  /*   while(1){} */
  /* } */
  /*
    if ( example_blink_init() ) {
    while(1){}
    }
  */

  /* blink_rgb(BLUE_LED, 1); */
  if ( tiqu_task_init() ) {
    while(1){}
  }

  if (thruster_task_init()) {
    while(1){}
  }

  /* if (qubobus_test_init() ){ */
  /*   while(1){} */
  /* } */
  /*
    if ( read_uart0_init() ) {
    while(1){}
    }
  */
  /*
    if( USB_serial_init() ){
    blink_rgb(RED_LED, 1);
    while(1){}
    }
  */
  /*
    if ( bme280_task_init()){
    while(1){}
    }
  */

  /*
    if ( example_uart_init() ) {
    while(1){}
    }
  */

  vTaskStartScheduler();

  while(1){
    blink_rgb(RED_LED, 1);
  }
}
