/*
Copyright 2014, Jernej Kovacic

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

/**
 * @file
 *
 * Definition of exception and interrupt vectors, implementation
 * of reset (and startup) handler and initialization of the BSS
 * section.
 *
 * For more info, see :
 * - Tiva(TM) TM4C123GH6PM Microcontroller Data Sheet
 * available at:
 * http://www.ti.com/lit/ds/symlink/tm4c123gh6pm.pdf
 *
 * and
 *
 * Cortex-M4 Devices, Generic User Guide (DUI553A)
 * available at:
 * http://infocenter.arm.com/help/topic/com.arm.doc.dui0553a/DUI0553A_cortex_m4_dgug.pdf
 *
 * @author Jernej Kovacic
 */


#include <stdint.h>
#include <stddef.h>

#include "FreeRTOSConfig.h"


/**
 * Required prototype for exception
 * and interrupt handler functions:
 */
typedef void (*NvicIsrType)(void);



/* Reserved space for the main stack */
static uint32_t mstack[ APP_MAIN_STACK_SIZE_WORDS ];

/* Reserved space for the process stack */
static uint32_t pstack[ APP_PROCESS_STACK_SIZE_WORDS ];


/* Initial process stack pointer: */
static const uint32_t* const _psp = pstack + APP_PROCESS_STACK_SIZE_WORDS;


/* Forward declarations: */
static void ResetISR(void);
static void NmiSR(void);
static void FaultISR(void);
static void IntDefaultHandler(void);


/* Implemented in init.c: */
extern void _init( void );
/* Implemented in main.c: */
extern int main(void);
/* Implemented in watchdog.c: */
extern void _wd_intHandler(void);


/* Implemented in port.c */
extern void xPortSysTickHandler(void);
extern void xPortPendSVHandler(void);
extern void vPortSVCHandler(void);

/* Implemented in handlers.c */
extern void GpioAIntHandler(void);
extern void GpioBIntHandler(void);
extern void GpioCIntHandler(void);
extern void GpioDIntHandler(void);
extern void GpioEIntHandler(void);
extern void GpioFIntHandler(void);

extern void Uart0IntHandler(void);
extern void Uart1IntHandler(void);
extern void Uart2IntHandler(void);
extern void Uart3IntHandler(void);
extern void Uart4IntHandler(void);
extern void Uart5IntHandler(void);
extern void Uart6IntHandler(void);
extern void Uart7IntHandler(void);


/*
 * An array with addresses of exception and interrupt handling
 * functions. Fore more information about the structure of the
 * vector table, see pp. 103 - 106 of the Data Sheet.
 *
 * Note: the __attribute__ decorator defines a special section,
 * the vector table belongs to. This will help the linker script
 * to place it to the correct location in the memory map.
 *
 * Note: although C does not strictly require it (anymore),
 * a '&' sign is prepended to all function addresses,
 * just to be aware, the array contains addresses.
 */
__attribute__ ((section(".isr_vector")))
static const NvicIsrType vectors[] =
{
    (NvicIsrType) ( (uint32_t) mstack + sizeof(mstack) ),
                                            /* The initial main stack pointer */
    &ResetISR,                              /* The reset handler              */
    &NmiSR,                                 /* The NMI handler                */
    &FaultISR,                              /* The hard fault handler         */
    &IntDefaultHandler,                     /* The MPU fault handler          */
    &IntDefaultHandler,                     /* The bus fault handler          */
    &IntDefaultHandler,                     /* The usage fault handler        */
    NULL,                                   /* Reserved                       */
    NULL,                                   /* Reserved                       */
    NULL,                                   /* Reserved                       */
    NULL,                                   /* Reserved                       */
    &vPortSVCHandler,                       /* SVCall handler                 */
    &IntDefaultHandler,                     /* Debug monitor handler          */
    NULL,                                   /* Reserved                       */
    &xPortPendSVHandler,                    /* The PendSV handler             */
    &xPortSysTickHandler,                   /* The SysTick handler            */
    &GpioAIntHandler,                       /* GPIO Port A                    */
    &GpioBIntHandler,                       /* GPIO Port B                    */
    &GpioCIntHandler,                       /* GPIO Port C                    */
    &GpioDIntHandler,                       /* GPIO Port D                    */
    &GpioEIntHandler,                       /* GPIO Port E                    */
    &Uart0IntHandler,                       /* UART0 Rx and Tx                */
    &Uart1IntHandler,                       /* UART1 Rx and Tx                */
    &IntDefaultHandler,                     /* SSI0 Rx and Tx                 */
    &IntDefaultHandler,                     /* I2C0 Master and Slave          */
    &IntDefaultHandler,                     /* PWM Fault                      */
    &IntDefaultHandler,                     /* PWM Generator 0                */
    &IntDefaultHandler,                     /* PWM Generator 1                */
    &IntDefaultHandler,                     /* PWM Generator 2                */
    &IntDefaultHandler,                     /* Quadrature Encoder 0           */
    &IntDefaultHandler,                     /* ADC Sequence 0                 */
    &IntDefaultHandler,                     /* ADC Sequence 1                 */
    &IntDefaultHandler,                     /* ADC Sequence 2                 */
    &IntDefaultHandler,                     /* ADC Sequence 3                 */
    &_wd_intHandler,                        /* Watchdog timer                 */
    &IntDefaultHandler,                     /* Timer 0 subtimer A             */
    &IntDefaultHandler,                     /* Timer 0 subtimer B             */
    &IntDefaultHandler,                     /* Timer 1 subtimer A             */
    &IntDefaultHandler,                     /* Timer 1 subtimer B             */
    &IntDefaultHandler,                     /* Timer 2 subtimer A             */
    &IntDefaultHandler,                     /* Timer 2 subtimer B             */
    &IntDefaultHandler,                     /* Analog Comparator 0            */
    &IntDefaultHandler,                     /* Analog Comparator 1            */
    &IntDefaultHandler,                     /* Analog Comparator 2            */
    &IntDefaultHandler,                     /* System Control (PLL, OSC, BO)  */
    &IntDefaultHandler,                     /* FLASH Control                  */
    &GpioFIntHandler,                       /* GPIO Port F                    */
    &IntDefaultHandler,                     /* GPIO Port G                    */
    &IntDefaultHandler,                     /* GPIO Port H                    */
    &Uart2IntHandler  ,                     /* UART2 Rx and Tx                */
    &IntDefaultHandler,                     /* SSI1 Rx and Tx                 */
    &IntDefaultHandler,                     /* Timer 3 subtimer A             */
    &IntDefaultHandler,                     /* Timer 3 subtimer B             */
    &IntDefaultHandler,                     /* I2C1 Master and Slave          */
    &IntDefaultHandler,                     /* Quadrature Encoder 1           */
    &IntDefaultHandler,                     /* CAN0                           */
    &IntDefaultHandler,                     /* CAN1                           */
    NULL,                                   /* Reserved                       */
    NULL,                                   /* Reserved                       */
    &IntDefaultHandler,                     /* Hibernate                      */
    &IntDefaultHandler,                     /* USB0                           */
    &IntDefaultHandler,                     /* PWM Generator 3                */
    &IntDefaultHandler,                     /* uDMA Software Transfer         */
    &IntDefaultHandler,                     /* uDMA Error                     */
    &IntDefaultHandler,                     /* ADC1 Sequence 0                */
    &IntDefaultHandler,                     /* ADC1 Sequence 1                */
    &IntDefaultHandler,                     /* ADC1 Sequence 2                */
    &IntDefaultHandler,                     /* ADC1 Sequence 3                */
    NULL,                                   /* Reserved                       */
    NULL,                                   /* Reserved                       */
    &IntDefaultHandler,                     /* GPIO Port J                    */
    &IntDefaultHandler,                     /* GPIO Port K                    */
    &IntDefaultHandler,                     /* GPIO Port L                    */
    &IntDefaultHandler,                     /* SSI2 Rx and Tx                 */
    &IntDefaultHandler,                     /* SSI3 Rx and Tx                 */
    &Uart3IntHandler,                       /* UART3 Rx and Tx                */
    &Uart4IntHandler,                       /* UART4 Rx and Tx                */
    &Uart5IntHandler,                       /* UART5 Rx and Tx                */
    &Uart6IntHandler,                       /* UART6 Rx and Tx                */
    &Uart7IntHandler,                       /* UART7 Rx and Tx                */
    NULL,                                   /* Reserved                       */
    NULL,                                   /* Reserved                       */
    NULL,                                   /* Reserved                       */
    NULL,                                   /* Reserved                       */
    &IntDefaultHandler,                     /* I2C2 Master and Slave          */
    &IntDefaultHandler,                     /* I2C3 Master and Slave          */
    &IntDefaultHandler,                     /* Timer 4 subtimer A             */
    &IntDefaultHandler,                     /* Timer 4 subtimer B             */
    NULL,                                   /* Reserved                       */
    NULL,                                   /* Reserved                       */
    NULL,                                   /* Reserved                       */
    NULL,                                   /* Reserved                       */
    NULL,                                   /* Reserved                       */
    NULL,                                   /* Reserved                       */
    NULL,                                   /* Reserved                       */
    NULL,                                   /* Reserved                       */
    NULL,                                   /* Reserved                       */
    NULL,                                   /* Reserved                       */
    NULL,                                   /* Reserved                       */
    NULL,                                   /* Reserved                       */
    NULL,                                   /* Reserved                       */
    NULL,                                   /* Reserved                       */
    NULL,                                   /* Reserved                       */
    NULL,                                   /* Reserved                       */
    NULL,                                   /* Reserved                       */
    NULL,                                   /* Reserved                       */
    NULL,                                   /* Reserved                       */
    NULL,                                   /* Reserved                       */
    &IntDefaultHandler,                     /* Timer 5 subtimer A             */
    &IntDefaultHandler,                     /* Timer 5 subtimer B             */
    &IntDefaultHandler,                     /* Wide Timer 0 subtimer A        */
    &IntDefaultHandler,                     /* Wide Timer 0 subtimer B        */
    &IntDefaultHandler,                     /* Wide Timer 1 subtimer A        */
    &IntDefaultHandler,                     /* Wide Timer 1 subtimer B        */
    &IntDefaultHandler,                     /* Wide Timer 2 subtimer A        */
    &IntDefaultHandler,                     /* Wide Timer 2 subtimer B        */
    &IntDefaultHandler,                     /* Wide Timer 3 subtimer A        */
    &IntDefaultHandler,                     /* Wide Timer 3 subtimer B        */
    &IntDefaultHandler,                     /* Wide Timer 4 subtimer A        */
    &IntDefaultHandler,                     /* Wide Timer 4 subtimer B        */
    &IntDefaultHandler,                     /* Wide Timer 5 subtimer A        */
    &IntDefaultHandler,                     /* Wide Timer 5 subtimer B        */
    &IntDefaultHandler,                     /* FPU                            */
    NULL,                                   /* Reserved                       */
    NULL,                                   /* Reserved                       */
    &IntDefaultHandler,                     /* I2C4 Master and Slave          */
    &IntDefaultHandler,                     /* I2C5 Master and Slave          */
    &IntDefaultHandler,                     /* GPIO Port M                    */
    &IntDefaultHandler,                     /* GPIO Port N                    */
    &IntDefaultHandler,                     /* Quadrature Encoder 2           */
    NULL,                                   /* Reserved                       */
    NULL,                                   /* Reserved                       */
    &IntDefaultHandler,                     /* GPIO Port P (Summary or P0)    */
    &IntDefaultHandler,                     /* GPIO Port P1                   */
    &IntDefaultHandler,                     /* GPIO Port P2                   */
    &IntDefaultHandler,                     /* GPIO Port P3                   */
    &IntDefaultHandler,                     /* GPIO Port P4                   */
    &IntDefaultHandler,                     /* GPIO Port P5                   */
    &IntDefaultHandler,                     /* GPIO Port P6                   */
    &IntDefaultHandler,                     /* GPIO Port P7                   */
    &IntDefaultHandler,                     /* GPIO Port Q (Summary or Q0)    */
    &IntDefaultHandler,                     /* GPIO Port Q1                   */
    &IntDefaultHandler,                     /* GPIO Port Q2                   */
    &IntDefaultHandler,                     /* GPIO Port Q3                   */
    &IntDefaultHandler,                     /* GPIO Port Q4                   */
    &IntDefaultHandler,                     /* GPIO Port Q5                   */
    &IntDefaultHandler,                     /* GPIO Port Q6                   */
    &IntDefaultHandler,                     /* GPIO Port Q7                   */
    &IntDefaultHandler,                     /* GPIO Port R                    */
    &IntDefaultHandler,                     /* GPIO Port S                    */
    &IntDefaultHandler,                     /* PWM 1 Generator 0              */
    &IntDefaultHandler,                     /* PWM 1 Generator 1              */
    &IntDefaultHandler,                     /* PWM 1 Generator 2              */
    &IntDefaultHandler,                     /* PWM 1 Generator 3              */
    &IntDefaultHandler                      /* PWM 1 Fault                    */
};

/*
 ******************************************************************************
 *
 * The following are constructs created by the linker, indicating where the
 * the "data" and "bss" segments reside in memory.  The initializers for the
 * for the "data" segment resides immediately following the "text" segment.
 *
 ******************************************************************************
 */

extern uint32_t _etext;
extern uint32_t _data;
extern uint32_t _edata;
extern uint32_t _bss;
extern uint32_t _ebss;


/*
 * Startup code, automatically executed on startup or reset.
 * It sets up the thread mode stack, initializes the BSS section,
 * copies data section initial values from flash to SRAM, and passes
 * control to the function main().
 *
 * The function is only executed on start up or reset, it never returns,
 * so there is no no need for any prologue and epilogue tasks and
 * wasting of stack with LR, etc., that is why the function
 * can be marked as "naked".
 */
__attribute__(( naked ))
static void ResetISR(void)
{
    /*
     * Do not declare any local variables until
     * the process stack is properly set up.
     */

    /*
     * This function is already started in Thread mode.
     * First the Control register will be set to use
     * the process stack.
     *
     * For more details about stacks, see page 2-2 of the DUI0553A
     * and page 74 of the Data Sheet.
     */

	/*
	 * Stack for the Thread Mode is selected by the ASP flag
	 * of the Control register.
	 *
	 * For more details about the register, see
	 * pp. 2-9 - 2.10 of the DUI0553A and
	 * pp. 88 - 89 of the Data Sheet.
	 */
    __asm volatile("    MRS     r0, control         ");  /* r0 = control      */
    __asm volatile("    ORR     r0, r0, #0x00000002 ");  /* r0 |= 2           */
    __asm volatile("    MSR     control, r0         ");  /* control = r0      */
    __asm volatile("    ISB                         ");  /* wait until synced */

    /*
     * After the Thread Mode stack has been set,
     * its stack pointer must be set.
     */
    __asm volatile("    LDR     r1, =_psp   ");   /* r1 = &_psp */
    __asm volatile("    LDR     r0, [r1]    ");   /* r0 = *r1   */
    __asm volatile("    MOV     sp, r0      ");   /* sp = r0    */
    __asm volatile("    ISB                 ");


    /*
     * Then initialize the BSS section.
     * Note that the BSS section may include the stack,
     * in this case initialization would also overwrite
     * local variables (in the stack), so the implementation
     * in C would probably not execute correctly. For this
     * reason, this task must be implemented in assembler.
     */

    __asm volatile("    LDR     r0, =_bss        ");  /* r0 = &_bss             */
    __asm volatile("    LDR     r1, =_ebss       ");  /* r1 = &_ebss            */
    __asm volatile("    MOV     r2, #0           ");  /* r2 = 0                 */
    __asm volatile("    .thumb_func              ");
    __asm volatile("bss_zero_loop:               ");
    __asm volatile("    CMP     r0, r1           ");  /* if (r0<r1)             */
    __asm volatile("    IT      lt               ");  /* {                      */
    __asm volatile("    STRLT   r2, [r0], #4     ");  /*   *(r0++) = r2         */
    __asm volatile("    BLT     bss_zero_loop    ");  /*   goto bss_zero_loop } */



    /*
     * Most likely the compiler will be able to
     * copy data initializers without pushing
     * these local variables to stack.
     */
    uint32_t* src;
    uint32_t* dest;

    /*
     * Copy the data segment initializers from flash to SRAM.
     */
    src = &_etext;
    for( dest = &_data; dest < &_edata; )
    {
        *dest++ = *src++;
    }

    /*
     * Suppress warnings, that _psp and vectors "are never used".
     * They actually are used in inline assembler code above
     * and by the linker script, respectively.
     */
    (void) _psp;
    (void) vectors;

    /* Initialize the MCU's peripherals: */
    _init();

    /*
     * The main function that continues the initialization and
     * starts the application:
     */
    main();

    /* An infinite loop. Just in case if main() ever returns... */
    for ( ; ; );
}


/*
 * Non-maskable interrupt handler
 */
__attribute__ ((interrupt))
static void NmiSR(void)
{
    /*
     * This interrupt is not supported yet, hence
     * end up in an infinite loop.
     */
    for ( ; ; );
}


/*
 * Hard Fault handler
 */
__attribute__ ((interrupt))
static void FaultISR(void)
{
    /*
     * This exception is not supported yet, hence
     * end up in an infinite loop.
     */
    for ( ; ; );
}


/*
 * Default interrupt handler, typically assigned
 * to non-configured IRQS.
 * It will end up in an infinite loop, signaling
 * that something unexpected has occurred.
 */
__attribute__ ((interrupt))
static void IntDefaultHandler(void)
{
    for ( ; ; );
}

