/** Ross Baehr
    Read info from UART0(part of the ICDI which can be used as a COM device)
    Put that info in a queue that another task is blocked on, so it will do shit with it
    Maybe have input determine what i2c device to get info from, which gets printed into UART
*/

#include "readinfo.h"

#include <inc/hw_memmap.h>
#include <inc/hw_ints.h>
#include <driverlib/interrupt.h>
#include <driverlib/gpio.h>
#include <driverlib/pin_map.h>
#include <driverlib/sysctl.h>
#include <driverlib/uart.h>
#include <utils/uartstdio.h>

void _readinfo_handler(void) {
  uint32_t status = UARTIntStatus(UART0_BASE, true);

  UARTIntClear(UART0_BASE, status);

  while (UARTCharsAvail(UART0_BASE)) {
    int32_t ch = UARTCharGet(UART0_BASE);

    // Send the string to the readinfo_task
    xQueueSendToBackFromISR(uart_input, &ch, NULL);
  }
}

  // msg is passed by value in the queue, but it's members have to be allocated in heap so they don't die
void readinfo_task(void* params) {

  uart_input = xQueueCreate(INPUT_BUFFER_SIZE, sizeof(int32_t));

  QueueHandle_t *processing_queue = (QueueHandle_t *) params;

  int32_t ch;
  readinfo_msg msg;

  int buffer_pos = 0;

  for (;;) {

    // Blocked until something appears in uart_input queue
    xQueueReceive(uart_input, (void*) &ch, portMAX_DELAY);

    switch (ch) {
      /* Uppercase letters 'A' .. 'Z': */
    case 'A' ... 'Z' :

      /* Lowercase letters 'a'..'z': */
    case 'a' ... 'z' :

      /* Decimal digits '0'..'9': */
    case '0' ... '9' :

      /* Other valid characters: */
    case ' ' :
    case '_' :
    case '+' :
    case '-' :
    case '/' :
    case '.' :
    case ',' :
      {
        if ( buffer_pos < INPUT_BUFFER_SIZE ) {
          msg.buffer[buffer_pos] = ch;
          ++buffer_pos;
        }
        break;
      }
    case CODE_BS :
      {
        /*
         * If the buffer is not empty, decrease the position index,
         * i.e. "delete" the last character
         */
        if ( buffer_pos>0 )
          {
            --buffer_pos;
          }
        break;
      }
    case CODE_CR :
      {
        /* Append characters to terminate the string:*/
        msg.buffer[buffer_pos] = '\0';
        /* Send the the string's pointer to the queue: */
        xQueueSendToBack(*processing_queue, (void*) &msg, 0);
        /* And switch to the next line of the "circular" buffer */
        buffer_pos = 0;
        break;
      }

    }
  }
}
