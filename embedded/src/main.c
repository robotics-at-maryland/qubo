/* Ross Baehr
   R@M 2017
   ross.baehr@gmail.com
*/

// Uncomment if want debug messages sent to uart. Requires linking to be done with tiva drivers
//#define DEBUG

//QSCU
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

  #ifdef DEBUG
  //
  // Initialize the UART for console I/O.
  //
  UARTStdioConfig(0, 115200, 16000000);
  #endif
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

void configureI2C(void) {
  //
  // Enable the peripherals used by this example.
  //
  ROM_SysCtlPeripheralEnable(SYSCTL_PERIPH_I2C0);
  ROM_SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOB);


  //
  // Configure the pin muxing for I2C0 functions on port B2 and B3.
  // This step is not necessary if your part does not support pin muxing.
  // TODO: change this to select the port/pin you are using.
  //
  ROM_GPIOPinConfigure(GPIO_PB2_I2C0SCL);
  ROM_GPIOPinConfigure(GPIO_PB3_I2C0SDA);

  // Select the I2C function for these pins.
  ROM_GPIOPinTypeI2CSCL(GPIO_PORTB_BASE, GPIO_PIN_2);
  ROM_GPIOPinTypeI2C(GPIO_PORTB_BASE, GPIO_PIN_3);


  //
  // Initialize the I2C master.
  //
  ROM_I2CMasterInitExpClk(I2C0_BASE, ROM_SysCtlClockGet(), false);

  //
  // Enable the I2C interrupt.
  //
  ROM_IntEnable(INT_I2C0);

  //
  // Enable the I2C master interrupt.
  //
  ROM_I2CMasterIntEnable(I2C0_BASE);
}


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

  // Master enable interrupts
  ROM_IntMasterEnable();

  configureUART();
  configureGPIO();
  configureI2C();

  #ifdef DEBUG
  UARTprintf("\n\nTask running\n");
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


  vTaskStartScheduler();

  while(1){}
}
