#ifndef _READUART_H_
#define _READUART_H_

// Triggered on a UART interrupt.
void _read_uart_handler(void);

void read_uart_task(void* params);

#endif
