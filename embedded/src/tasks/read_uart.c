/**
   Interrupt generates dynamically allocated buffer of messages.
   Their pointers are added to the read_uart task queue, who
   decides what to do with them
 */

#include "tasks/include/read_uart.h"

// For testing purposes
#include "lib/include/write_uart.h"

bool read_uart_init(void) {
  read_uart = xQueueCreate(Q_SIZE, sizeof(uint8_t));

  if ( xTaskCreate(read_uart_task, (const portCHAR *)"Read UART", 128, NULL,
                   tskIDLE_PRIORITY + 1, NULL) != pdTRUE) {
    return true;
  }
  return false;
}

void UARTIntHandler(void) {
  uint32_t status = ROM_UARTIntStatus(UART_DEVICE, true);

  // Clear interrupt
  ROM_UARTIntClear(UART_DEVICE, status);

  if ( !ROM_UARTCharsAvail(UART_DEVICE) ) {
    // ERROR, Do something here
  }

  // Get one byte
  // Tivaware casts the byte to a int32_t for some reason, cast back to save space
  uint8_t c = (uint8_t)(ROM_UARTCharGetNonBlocking(UART_DEVICE));
  // Push to the queue
  xQueueSendToBackFromISR(read_uart, c, NULL);

  }


void read_uart_task(void* params) {

  // Qubobus driver code to assemble/interpret messages here
  uint8_t buffer[8] = "Recieved";

  for (;;) {

    if ( xQueueReceive(read_uart, buffer, 0) == pdTRUE ) {
      writeUART(&buffer, 8);
    }
  }
}
