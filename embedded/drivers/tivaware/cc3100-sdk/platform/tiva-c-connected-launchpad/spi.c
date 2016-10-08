/*
 * spi.c - tiva-c-connected launchpad spi interface implementation
 *
 * Copyright (C) 2014 Texas Instruments Incorporated - http://www.ti.com/
 *
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *    Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 *    Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the
 *    distribution.
 *
 *    Neither the name of Texas Instruments Incorporated nor the names of
 *    its contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
*/

#include "simplelink.h"
#include "board.h"
#include "spi.h"
#include "inc/hw_memmap.h"
#include "inc/hw_ssi.h"
#include "inc/hw_gpio.h"
#include "inc/hw_types.h"
#include "inc/hw_ints.h"
#include "driverlib/ssi.h"
#include "driverlib/rom.h"
#include "driverlib/gpio.h"
#include "driverlib/sysctl.h"
#include "driverlib/interrupt.h"
#include "driverlib/pin_map.h"


#define ASSERT_CS()         GPIOPinWrite(GPIO_PORTP_BASE,GPIO_PIN_5, PIN_LOW)
#define DEASSERT_CS()       GPIOPinWrite(GPIO_PORTP_BASE,GPIO_PIN_5, PIN_HIGH)

extern _u32 g_SysClock;

int spi_Close(Fd_t fd)
{
    /* Disable WLAN Interrupt ... */
    CC3100_InterruptDisable();

    return NONOS_RET_OK;
}

Fd_t spi_Open(char *ifName, unsigned long flags)
{
    /* Configure CS (PP5) and nHIB (PD4) lines */
    SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOP);
    ROM_GPIOPinTypeGPIOOutput(GPIO_PORTP_BASE, GPIO_PIN_5);
    ROM_GPIOPinWrite(GPIO_PORTP_BASE,GPIO_PIN_5, PIN_HIGH);

    SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOD);
    ROM_GPIOPinTypeGPIOOutput(GPIO_PORTD_BASE, GPIO_PIN_4);
    ROM_GPIOPinWrite(GPIO_PORTD_BASE,GPIO_PIN_4, PIN_LOW);

    SysCtlPeripheralEnable(SYSCTL_PERIPH_SSI3);
    SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOQ);

    GPIOPinConfigure(GPIO_PQ0_SSI3CLK);   //CLK
    GPIOPinConfigure(GPIO_PQ2_SSI3XDAT0); //MOSI
    GPIOPinConfigure(GPIO_PQ3_SSI3XDAT1); //MISO

    GPIOPinTypeSSI(GPIO_PORTQ_BASE, GPIO_PIN_0 | GPIO_PIN_2
            | GPIO_PIN_3);

    SSIConfigSetExpClk(SSI3_BASE,  g_SysClock, SSI_FRF_MOTO_MODE_0,
                SSI_MODE_MASTER, 12000000, 8);

    SSIEnable(SSI3_BASE);

    /* configure host IRQ line */
    SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOM);
    GPIOIntDisable(GPIO_PORTM_BASE, GPIO_PIN_7);

    GPIOPinTypeGPIOInput(GPIO_PORTM_BASE, GPIO_PIN_7);
    GPIOPadConfigSet(GPIO_PORTM_BASE, GPIO_PIN_7, GPIO_STRENGTH_2MA,
            GPIO_PIN_TYPE_STD_WPD);
    GPIOIntTypeSet(GPIO_PORTM_BASE, GPIO_PIN_7, GPIO_RISING_EDGE);

    GPIOIntStatus(GPIO_PORTM_BASE, 1);
    GPIOIntClear(GPIO_PORTM_BASE,GPIO_PIN_7);
    GPIOIntEnable(GPIO_PORTM_BASE,GPIO_PIN_7);

    IntEnable(INT_GPIOM);
    IntMasterEnable();

    SysCtlDelay(g_SysClock * 20 / (3 * 1000));

    /* Enable WLAN interrupt */
    CC3100_InterruptEnable();

    return NONOS_RET_OK;
}


int spi_Write(Fd_t fd, unsigned char *pBuff, int len)
{
    int len_to_return = len;
    unsigned long ulDummy;

    ASSERT_CS();

    while(len)
    {
        while(SSIDataPutNonBlocking(SSI3_BASE, (unsigned long)*pBuff) != TRUE);
        while(SSIDataGetNonBlocking(SSI3_BASE, &ulDummy) != TRUE);
        pBuff++;
        len--;
    }

    while(SSIBusy(SSI3_BASE));

    DEASSERT_CS();

    return len_to_return;
}


int spi_Read(Fd_t fd, unsigned char *pBuff, int len)
{
    int i = 0;
    unsigned long ulBuff;

    ASSERT_CS();

    for(i=0; i< len; i++)
    {
        while(SSIDataPutNonBlocking(SSI3_BASE, 0xFF) != TRUE);
        while(SSIDataGetNonBlocking(SSI3_BASE, &ulBuff) != TRUE);
        pBuff[i] = (unsigned char)ulBuff;
    }

    while(SSIBusy(SSI3_BASE));

    DEASSERT_CS();

    return len;
}
