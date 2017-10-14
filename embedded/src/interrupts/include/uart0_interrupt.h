/* Ross Baehr
   R@M 2017
   ross.baehr@gmail.com

   Greg Harris
   gharris1727@gmail.com
*/

#ifndef _UART0_INTERRUPT_H_
#define _UART0_INTERRUPT_H_

extern volatile struct UART_Queue uart0_queue;

void UART0IntHandler(void);

#endif
