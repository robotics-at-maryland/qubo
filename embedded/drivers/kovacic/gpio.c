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
 * Implementation of General-Purpose Input/Output (GPIO)
 * functionality. All 6 GPIO ports are supported.
 *
 * Note: Advanced High-Performance Bus (AHB) must be enabled
 *       before any GPIO register is accessed!
 *
 * For more info about the GPIO port controllers,
 * see pp. 649 - 703 of:
 * Tiva(TM) TM4C123GH6PM Microcontroller Data Sheet,
 * available at:
 * http://www.ti.com/lit/ds/symlink/tm4c123gh6pm.pdf
 *
 * @author Jernej Kovacic
 */


#include <stdint.h>
#include <stddef.h>

#include "FreeRTOSConfig.h"
#include "bsp.h"
#include "gpio.h"
#include "regutil.h"
#include "sysctl.h"
#include "nvic.h"


/*
 * 32-bit Registers of individual GPIO port controllers,
 * relative to the controller's base address:
 * See pages 658 - 661 of the Data Sheet.
 */
typedef struct _TM4C123G_GPIO_REGS
{
    /*
    According to the register map (pp. 658 -661 of theData Sheet),
    the struct should start like this:

    uint32_t GPIO_DATA;                // GPIO Data
    const uint32_t Reserved1[255];     // reserved

    However, access to DATA register's pins is more tricky,
    hence an array of size 256 words is introduced instead:
    */
    uint32_t ADDR[256];

    /* past this point, register layout matches the register map: */
    uint32_t GPIO_DIR;                 /* GPIO Direction */
    uint32_t GPIO_IS;                  /* GPIO Interrupt Sense */
    uint32_t GPIO_IBE;                 /* GPIO Interrupt Both Edges */
    uint32_t GPIO_IEV;                 /* GPIO Interrupt Event */
    uint32_t GPIO_IM;                  /* GPIO Interrupt Mask */
    const uint32_t GPIO_RIS;           /* GPIO Raw Interrupt Status, read only */
    const uint32_t GPIO_MIS;           /* GPIO Masked Interrupt Status, read only */
    uint32_t GPIO_ICR;                 /* GPIO Interrupt Clear, write only */
    uint32_t GPIO_AFSEL;               /* GPIO Alternate Function Select */
    const uint32_t Reserved2[55];      /* reserved */
    uint32_t GPIO_DR2R;                /* GPIO 2-mA Drive Select */
    uint32_t GPIO_DR4R;                /* GPIO 4-mA Drive Select */
    uint32_t GPIO_DR8R;                /* GPIO 8-mA Drive Select */
    uint32_t GPIO_ODR;                 /* GPIO Open Drain Select */
    uint32_t GPIO_PUR;                 /* GPIO Pull-Up Select */
    uint32_t GPIO_PDR;                 /* GPIO Pull-Down Select */
    uint32_t GPIO_SLR;                 /* GPIO Slew Rate Control Select */
    uint32_t GPIO_DEN;                 /* GPIO Digital Enable */
    uint32_t GPIO_LOCK;                /* GPIO Lock */
    uint32_t GPIO_CR;                  /* GPIO Commit */
    uint32_t GPIO_AMSEL;               /* GPIO Analog Mode Select */
    uint32_t GPIO_PCTL;                /* GPIO Port Control */
    uint32_t GPIO_ADCCTL;              /* GPIO ADC Control */
    uint32_t GPIO_DMACTL;              /* GPIO DMA Control */
    const uint32_t Reserved3[678];     /* reserved */
    const uint32_t GPIO_PeriphID4;     /* GPIO Peripheral Identification 4, read only */
    const uint32_t GPIO_PeriphID5;     /* GPIO Peripheral Identification 5, read only */
    const uint32_t GPIO_PeriphID6;     /* GPIO Peripheral Identification 6, read only */
    const uint32_t GPIO_PeriphID7;     /* GPIO Peripheral Identification 7, read only */
    const uint32_t GPIO_PeriphID0;     /* GPIO Peripheral Identification 0, read only */
    const uint32_t GPIO_PeriphID1;     /* GPIO Peripheral Identification 1, read only */
    const uint32_t GPIO_PeriphID2;     /* GPIO Peripheral Identification 2, read only */
    const uint32_t GPIO_PeriphID3;     /* GPIO Peripheral Identification 3, read only */
    const uint32_t GPIO_CellID0;       /* GPIO PrimeCell Identification 0, read only */
    const uint32_t GPIO_CellID1;       /* GPIO PrimeCell Identification 1, read only */
    const uint32_t GPIO_CellID2;       /* GPIO PrimeCell Identification 2, read only */
    const uint32_t GPIO_CellID3;       /* GPIO PrimeCell Identification 3, read only */

} TM4C123G_GPIO_REGS;


/* --------------------------------------------------- */
#define GEN_CAST_ADDR(ADDR)         (volatile TM4C123G_GPIO_REGS* const) (ADDR),

static volatile TM4C123G_GPIO_REGS* const pReg[ BSP_NR_GPIO_PORTS ]=
{

#if APP_GPIO_AHB != 0
    BSP_GPIO_BASE_ADDRESSES_AHB( GEN_CAST_ADDR )
#else
    BSP_GPIO_BASE_ADDRESSES_APB( GEN_CAST_ADDR )
#endif

};

#undef GEN_CAST_ADDR
/* --------------------------------------------------- */


/* IRQs for each port */
static const uint8_t __gpioIrqs[ BSP_NR_GPIO_PORTS ] = BSP_GPIO_IRQS;


/* Nr. of pins, controlled by one GPIO port: */
#define PINS_PER_PORT      ( 8 )

/* Maximum value for one pin's PCTL: */
#define PCTL_MAX_VALUE     ( 15 )

/* Convenience bit mask for setting of PCTL values: */
#define PCTL_MASK          ( 0x0000000F )

/* A constant for unlocking the GPIOCR register: */
#define PORT_UNLOCK        ( 0x4C4F434B )


/*
 * Table of interrupt handling routines for each GPIO port and pin
 */
static GpioPortIntHandler_t __intrIsr[ BSP_NR_GPIO_PORTS ][ PINS_PER_PORT ];


/*
 * A dummy GPIO interrupt "handler", executed when
 * no handler has been registered to handle interrupt
 * requests from a certain port and pin.
 *
 * This function is set as default to __intrIsr table.
 */
static void __defaultIntHandler(void)
{
   /* intentionally left blank */
}


/**
 * Enables the selected GPIO port at the System Controller.
 * When the port is enabled, it is provided a clock and
 * access to its registers is allowed.
 *
 * Nothing is done if 'port' is greater than 5.
 *
 * @param port - GPIO port number to be enabled (between 0 and 5)
 */
void gpio_enablePort(uint8_t port)
{
    if ( port < BSP_NR_GPIO_PORTS )
    {
        sysctl_enableGpioPort(port);
    }
}


/**
 * Disables the selected GPIO port at the System Controller.
 * When the port is disabled, it is disconnected from the clock
 * and access to its registers is not allowed. Any attempt of
 * accessing a disabled port may result in a bus fault.
 *
 * Nothing is done if 'port' is greater than 5.
 *
 * @param port - GPIO port number to be disabled (between 0 and 5)
 */
void gpio_disablePort(uint8_t port)
{
    if ( port < BSP_NR_GPIO_PORTS )
    {
        sysctl_disableGpioPort(port);
    }
}


/**
 * Configures the selected pin as an input pin.
 *
 * Nothing is done if 'port' is greater than 5 or
 * 'pin' is greater than 7.
 *
 * @param port - desired GPIO port number (between 0 and 5)
 * @param pin - pin on the selected GPIO port (between 0 and 7)
 */
void gpio_setPinAsInput(uint8_t port, uint8_t pin)
{
    /*
     * See the Data Sheet, page 663, for more details.
     */

    if (port<BSP_NR_GPIO_PORTS && pin<PINS_PER_PORT )
    {
        HWREG_CLEAR_SINGLE_BIT( pReg[port]->GPIO_DIR, pin );
    }
}


/**
 * Configures the selected pin as an output pin.
 *
 * Nothing is done if 'port' is greater than 5 or
 * 'pin' is greater than 7.
 *
 * @param port - desired GPIO port number (between 0 and 5)
 * @param pin - pin on the selected GPIO port (between 0 and 7)
 */
void gpio_setPinAsOutput(uint8_t port, uint8_t pin)
{
    /*
     * See the Data Sheet, page 663, for more details.
     */

    if (port<BSP_NR_GPIO_PORTS && pin<PINS_PER_PORT )
    {
        HWREG_SET_SINGLE_BIT( pReg[port]->GPIO_DIR, pin );
    }
}


/**
 * Selects whether the selected pin will be controlled by GPIO
 * registers or a peripheral. In the later case it also selects
 * the peripheral signal for the pin.
 *
 * See Table 23-5 (pages 1352 - 1353) in the Data Sheet for
 * more details, which peripheral signal can be assigned to
 * certain pins.
 *
 * Nothing is done if 'port' is greater than 5 or
 * 'pin' is greater than 7 or 'pctl' is greater than 15.
 *
 * @param port - desired GPIO port number (between 0 and 5)
 * @param pin - pin on the selected GPIO port (between 0 and 7)
 * @param pctl - function for the pin, 0 for GPIO, anything else for a peripheral (between 0 and 15)
 */
void gpio_setAltFunction(uint8_t port, uint8_t pin, uint8_t pctl)
{
    /* See the Data Sheet, pages 671 and 688 - 689, for more details. */

    if ( port < BSP_NR_GPIO_PORTS &&
         pin < PINS_PER_PORT &&
         pctl <= PCTL_MAX_VALUE )
    {
        if ( 0 == pctl )
        {
            /* configure as a GPIO pin: */
            HWREG_CLEAR_SINGLE_BIT( pReg[port]->GPIO_AFSEL, pin );
        }
        else
        {
            /* configure as an alternate function: */
            HWREG_SET_SINGLE_BIT( pReg[port]->GPIO_AFSEL, pin );
        }

        /* set PCTL regardless of the function: */
        HWREG_SET_CLEAR_BITS( pReg[port]->GPIO_PCTL, pctl << (4*pin), PCTL_MASK << (4*pin));
    }
}


/**
 * Enables the digital functions for the selected pin.
 * If this is not enabled, the pin does not drive a
 * logic value on the pin and it does not allow the pin voltage
 * into the GPIO receiver.
 *
 * Nothing is done if 'port' is greater than 5 or
 * 'pin' is greater than 7.
 *
 * @param port - desired GPIO port number (between 0 and 5)
 * @param pin - pin on the selected GPIO port (between 0 and 7)
 */
void gpio_enableDigital(uint8_t port, uint8_t pin)
{
    /*
     * See the Data Sheet, pages 682 - 683, for more details.
     */
    if ( port<BSP_NR_GPIO_PORTS && pin<PINS_PER_PORT )
    {
        HWREG_SET_SINGLE_BIT( pReg[port]->GPIO_DEN, pin );
    }
}


/**
 * Disables the digital functions for the selected pin.
 * If this is disabled, the pin does not drive a
 * logic value on the pin and it does not allow the pin voltage
 * into the GPIO receiver.
 *
 * Nothing is done if 'port' is greater than 5 or
 * 'pin' is greater than 7.
 *
 * @param port - desired GPIO port number (between 0 and 5)
 * @param pin - pin on the selected GPIO port (between 0 and 7)
 */
void gpio_disableDigital(uint8_t port, uint8_t pin)
{
    /*
     * See the Data Sheet, pages 682 - 683, for more details.
     */

    if ( port<BSP_NR_GPIO_PORTS && pin<PINS_PER_PORT )
    {
        HWREG_CLEAR_SINGLE_BIT( pReg[port]->GPIO_DEN, pin );
    }
}


/**
 * Enables analog mode for the selected pin.
 *
 * When this is enabled, the pin isolation is disabled
 * and the pin is capable of analog functions.
 *
 * @param port - desired GPIO port number (between 0 and 5)
 * @param pin - pin on the selected GPIO port (between 0 and 7)
 */
void gpio_enableAnalog(uint8_t port, uint8_t pin)
{
    /*
     * See the Data Sheet, page 687, for more details.
     */

    if ( port<BSP_NR_GPIO_PORTS && pin<PINS_PER_PORT )
    {
        HWREG_SET_SINGLE_BIT( pReg[port]->GPIO_AMSEL, pin );
    }
}


/**
 * Disables analog mode for the selected pin.
 *
 * When this is disabled, the pin isolation is enabled
 * and the pin is capable of digital functions as specified
 * by the other GPIO configuration registers.
 *
 * @param port - desired GPIO port number (between 0 and 5)
 * @param pin - pin on the selected GPIO port (between 0 and 7)
 */
void gpio_disableAnalog(uint8_t port, uint8_t pin)
{
    /*
     * See the Data Sheet, page 687, for more details.
     */

    if ( port<BSP_NR_GPIO_PORTS && pin<PINS_PER_PORT )
    {
        HWREG_CLEAR_SINGLE_BIT( pReg[port]->GPIO_AMSEL, pin );
    }
}


/**
 * Enables a weak pull-up resistor on the selected pin.
 *
 * Nothing is done if 'port' is greater than 5 or
 * 'pin' is greater than 7.
 *
 * @param port - desired GPIO port number (between 0 and 5)
 * @param pin - pin on the selected GPIO port (between 0 and 7)
 */
void gpio_enablePullUp(uint8_t port, uint8_t pin)
{
    /*
     * See the Data Sheet, pages 677 - 678, for more details.
     */
    if ( port<BSP_NR_GPIO_PORTS && pin<PINS_PER_PORT )
    {
        HWREG_SET_SINGLE_BIT( pReg[port]->GPIO_PUR, pin);
    }
}


/**
 * Disables a weak pull-up resistor on the selected pin.
 *
 * Nothing is done if 'port' is greater than 5 or
 * 'pin' is greater than 7.
 *
 * @param port - desired GPIO port number (between 0 and 5)
 * @param pin - pin on the selected GPIO port (between 0 and 7)
 */
void gpio_disablePullUp(uint8_t port, uint8_t pin)
{
    /*
     * See the Data Sheet, pages 677 - 678, for more details.
     */

    if ( port<BSP_NR_GPIO_PORTS && pin<PINS_PER_PORT )
    {
        HWREG_CLEAR_SINGLE_BIT( pReg[port]->GPIO_PUR, pin );
    }
}


/**
 * Enables a weak pull-down resistor on the selected pin.
 *
 * Nothing is done if 'port' is greater than 5 or
 * 'pin' is greater than 7.
 *
 * @param port - desired GPIO port number (between 0 and 5)
 * @param pin - pin on the selected GPIO port (between 0 and 7)
 */
void gpio_enablePullDown(uint8_t port, uint8_t pin)
{
    /*
     * See the Data Sheet, pages 679 - 680, for more details.
     */

    if ( port<BSP_NR_GPIO_PORTS && pin<PINS_PER_PORT )
    {
        HWREG_SET_SINGLE_BIT( pReg[port]->GPIO_PDR, pin );
    }
}


/**
 * Disables a weak pull-down resistor on the selected pin.
 *
 * Nothing is done if 'port' is greater than 5 or
 * 'pin' is greater than 7.
 *
 * @param port - desired GPIO port number (between 0 and 5)
 * @param pin - pin on the selected GPIO port (between 0 and 7)
 */
void gpio_disablePullDown(uint8_t port, uint8_t pin)
{
    /*
     * See the Data Sheet, pages 679 - 680, for more details.
     */

    if ( port<BSP_NR_GPIO_PORTS && pin<PINS_PER_PORT )
    {
        HWREG_CLEAR_SINGLE_BIT( pReg[port]->GPIO_PDR, pin );
    }
}


/**
 * Enables the open-drain configuration for the selected pin.
 *
 * Nothing is done if 'port' is greater than 5 or
 * 'pin' is greater than 7.
 *
 * @param port - desired GPIO port number (between 0 and 5)
 * @param pin - pin on the selected GPIO port (between 0 and 7)
 */
void gpio_enableOpenDrain(uint8_t port, uint8_t pin)
{
    /*
     * See the Data Sheet, page 676, for more details.
     */
    if ( port<BSP_NR_GPIO_PORTS && pin<PINS_PER_PORT )
    {
        HWREG_SET_SINGLE_BIT( pReg[port]->GPIO_ODR, pin );
    }
}


/**
 * Disables the open-drain configuration for the selected pin.
 *
 * Nothing is done if 'port' is greater than 5 or
 * 'pin' is greater than 7.
 *
 * @param port - desired GPIO port number (between 0 and 5)
 * @param pin - pin on the selected GPIO port (between 0 and 7)
 */
void gpio_disableOpenDrain(uint8_t port, uint8_t pin)
{
    /*
     * See the Data Sheet, page 676, for more details.
     */

    if ( port<BSP_NR_GPIO_PORTS && pin<PINS_PER_PORT )
    {
        HWREG_CLEAR_SINGLE_BIT( pReg[port]->GPIO_ODR, pin );
    }
}


/**
 * Enables the 2 mA, 4 mA or 8 mA drive strength for the selected pin.
 *
 * Nothing is done if 'port' is greater than 5 or
 * 'pin' is greater than 7 or 'dr' is invalid.
 *
 * @param port - desired GPIO port number (between 0 and 5)
 * @param pin - pin on the selected GPIO port (between 0 and 7)
 * @param dr - desired drive strength for the pin
 */
void gpio_setDriveStrength(uint8_t port, uint8_t pin, gpio_drive_t dr)
{
    /*
     * See the Data Sheet, pages 673, 674 and 675, for more details.
     */

    if ( port<BSP_NR_GPIO_PORTS && pin<PINS_PER_PORT )
    {
        switch (dr)
        {
        case DR_2_MA:
            HWREG_SET_SINGLE_BIT( pReg[port]->GPIO_DR2R, pin );
            break;

        case DR_4_MA:
            HWREG_SET_SINGLE_BIT( pReg[port]->GPIO_DR4R, pin );
            break;

        case DR_8_MA:
            HWREG_SET_SINGLE_BIT( pReg[port]->GPIO_DR8R, pin );
            break;
        }
    }
}


/**
 * Sets the selected pin to 1.
 *
 * Nothing is done if 'port' is greater than 5 or
 * 'pin' is greater than 7.
 *
 * @param port - desired GPIO port number (between 0 and 5)
 * @param pin - pin on the selected GPIO port (between 0 and 7)
 */
void gpio_setPin(uint8_t port, uint8_t pin)
{
    /*
     * For detailed instructions about setting a GPIO pin,
     * see page 654 of the Data Sheet.
     *
     * Instead of accessing the GPIODATA register directly, a bit mask
     * for the selected pin(s) must be written to the address of the
     * same value, relative to the GPIO port controller's base address.
     * For that reason, an array of DATA "registers" was introduced.
     */

    if ( port<BSP_NR_GPIO_PORTS && pin<PINS_PER_PORT )
    {
        pReg[port]->ADDR[HWREG_SINGLE_BIT_MASK(pin)] = HWREG_SINGLE_BIT_MASK(pin);
    }
}

/**
 * Clears the selected pin to 0.
 *
 * Nothing is done if 'port' is greater than 5 or
 * 'pin' is greater than 7.
 *
 * @param port - desired GPIO port number (between 0 and 5)
 * @param pin - pin on the selected GPIO port (between 0 and 7)
 */
void gpio_clearPin(uint8_t port, uint8_t pin)
{
    /*
     * For detailed instructions about setting a GPIO pin,
     * see page 654 of the Data Sheet.
     *
     * Instead of accessing the GPIODATA register directly, a bit mask
     * for the selected pin(s) must be written to the address of the
     * same value, relative to the GPIO port controller's base address.
     * For that reason, an array of DATA "registers" was introduced.
     */

    if ( port<BSP_NR_GPIO_PORTS && pin<PINS_PER_PORT )
    {
        pReg[port]->ADDR[HWREG_SINGLE_BIT_MASK(pin)] = ~HWREG_SINGLE_BIT_MASK(pin);
    }
}


/**
 * Sets the selected pins of the same GPIO port to 1.
 *
 * The higher 24 pins of 'pinmask' will be discarded.
 *
 * Nothing is done if 'port' is greater than 5.
 *
 * @param port - desired GPIO port number (between 0 and 5)
 * @param pinmask - bit mask on the selected GPIO port to be set (between 0 and 255)
 */
void gpio_setPins(uint8_t port, uint8_t pinmask)
{
    /*
     * For detailed instructions about setting a GPIO pin,
     * see page 654 of the Data Sheet.
     *
     * Instead of accessing the GPIODATA register directly, a bit mask
     * for the selected pin(s) must be written to the address of the
     * same value, relative to the GPIO port controller's base address.
     * For that reason, an array of DATA "registers" was introduced.
     */

    if ( port < BSP_NR_GPIO_PORTS )
    {
        pReg[port]->ADDR[pinmask & 0x000000FF] = (pinmask & 0x000000FF);
    }
}


/**
 * Clears the selected pins of the same GPIO port to 0.
 *
 * The higher 24 pins of 'pinmask' will be discarded.
 *
 * Nothing is done if 'port' is greater than 5.
 *
 * @param port - desired GPIO port number (between 0 and 5)
 * @param pinmask - bit mask on the selected GPIO port to be cleared (between 0 and 255)
 */
void gpio_clearPins(uint8_t port, uint8_t pinmask)
{
    /*
     * For detailed instructions about setting a GPIO pin,
     * see page 654 of the Data Sheet.
     *
     * Instead of accessing the GPIODATA register directly, a bit mask
     * for the selected pin(s) must be written to the address of the
     * same value, relative to the GPIO port controller's base address.
     * For that reason, an array of DATA "registers" was introduced.
     */

    if ( port < BSP_NR_GPIO_PORTS )
    {
        pReg[port]->ADDR[pinmask & 0x000000FF] = ~(pinmask & 0x000000FF);
    }
}


/**
 * Reads the status of the selected pin.
 *
 * 0 is returned if 'port' is greater than 5 or
 * 'pin' is greater than 7.
 *
 * @param port - desired GPIO port number (between 0 and 5)
 * @param pin - pin on the selected GPIO port (between 0 and 7)
 *
 * @return 1 if the selected pin is set, 0 if it is not
 */
uint8_t gpio_readPin(uint8_t port, uint8_t pin)
{
    /*
     * For detailed instructions about reading the status of
     * a GPIO pin, see page 654 of the Data Sheet.
     *
     * Instead of accessing the GPIODATA register directly, an 8-bit value
     * for the selected pin(s) must be read from the address of the
     * the bit mask value, relative to the GPIO port controller's base address.
     * Then the desired pin statuses must be obtained from that value,
     * using the bitwise AND operator.
     * For that reason, an array of DATA "registers" was introduced.
     */

    return
        ( port<BSP_NR_GPIO_PORTS && pin<PINS_PER_PORT &&
          0 != ( HWREG_READ_SINGLE_BIT(pReg[port]->ADDR[MASK_ONE << pin], pin) ) ?
          1 : 0 );
}


/**
 * Reads the statuses of the selected pins on the same GPIO port.
 * Only statuses of pins specified by 'pinmask' will be read,
 * statuses of all other bits will be returned as 0.
 *
 * To read a status of individual bit(s), a bitwise AND must
 * be performed on the return value.
 *
 * The higher 24 pins of 'pinmask' will be discarded.
 *
 * 0 is returned if 'port' is greater than 5.
 *
 * @param port - desired GPIO port number (between 0 and 5)
 * @param pinmask - bit mask on the selected GPIO port to be read (between 0 and 255)
 *
 * @return a bit mask with statuses of all pins (non-selected pins will be read as 0)
 */
uint8_t gpio_readPins(uint8_t port, uint8_t pinmask)
{
    /*
     * For detailed instructions about reading the status of
     * a GPIO pin, see page 654 of the Data Sheet.
     *
     * Instead of accessing the GPIODATA register directly, an 8-bit value
     * for the selected pin(s) must be read from the address of the
     * the bit mask value, relative to the GPIO port controller's base address.
     * Then the desired pin statuses must be obtained from that value,
     * using the bitwise AND operator.
     * For that reason, an array of DATA "registers" was introduced.
     */

    return
        ( port < BSP_NR_GPIO_PORTS ?
          (uint8_t) pReg[port]->ADDR[ pinmask & 0x000000FF ] : 0 );
}


/**
 * Enables the NVIC to process interrupt requests,
 * triggered by any pin of the selected port
 * (provided that interrupt requests for that pin
 * have not been masked out).
 *
 * Nothing is done if 'port' is greater than 5.
 *
 * @param port - desired GPIO port number (between 0 and 5)
 */
void gpio_enableInterruptPort(uint8_t port)
{
    if ( port < BSP_NR_GPIO_PORTS )
    {
        nvic_enableInterrupt( __gpioIrqs[port] );
    }
}


/**
 * Disables the NVIC to process interrupt requests,
 * triggered by any pin of the selected port.
 *
 * Nothing is done if 'port' is greater than 5.
 *
 * @param port - desired GPIO port number (between 0 and 5)
 */
void gpio_disableInterruptPort(uint8_t port)
{
    if ( port < BSP_NR_GPIO_PORTS )
    {
        nvic_disableInterrupt( __gpioIrqs[port] );
    }

}


/**
 * Sets priority level for interrupt requests, triggered
 * by the selected port.
 *
 * Nothing is done if 'port' is greater than 5
 * or 'pri' is greater than 7.
 *
 * @param port - desired GPIO port number (between 0 and 5)
 * @param pri - priority for interrupt requests from the port (between 0 and 7)
 */
void gpio_setIntrPriority(uint8_t port, uint8_t pri)
{
    if ( port < BSP_NR_GPIO_PORTS && pri <= MAX_PRIORITY )
    {
        nvic_setPriority( __gpioIrqs[port], pri );
    }
}


/**
 * Unmasks interrupts (i.e. they are sent to the NVIC controller),
 * triggered by the selected pin on the selected GPIO port.
 *
 * Nothing is done if 'port' is greater than 5 or
 * 'pin' is greater than 7.
 *
 * @param port - desired GPIO port number (between 0 and 5)
 * @param pin - pin on the selected GPIO port (between 0 and 7)
 */
void gpio_unmaskInterrupt(uint8_t port, uint8_t pin)
{
   /*
    * Unmasking of interrupts, triggered by events on
    * individual pins, is accomplished by setting
    * corresponding bits of the GPIOIM register to 1.
    *
    * For more detail, see page 667 of the Data Sheet.
    */

   if ( port < BSP_NR_GPIO_PORTS && pin < PINS_PER_PORT )
   {
       HWREG_SET_SINGLE_BIT( pReg[port]->GPIO_IM, pin );
   }
}


/**
 * Masks interrupts (i.e. they are not sent to the NVIC controller),
 * triggered by the selected pin on the selected GPIO port.
 *
 * Nothing is done if 'port' is greater than 5 or
 * 'pin' is greater than 7.
 *
 * @param port - desired GPIO port number (between 0 and 5)
 * @param pin - pin on the selected GPIO port (between 0 and 7)
 */
void gpio_maskInterrupt(uint8_t port, uint8_t pin)
{
    /*
     * Masking of interrupts, triggered by events on
     * individual pins, is accomplished by clearing
     * corresponding bits of the GPIOIM register to 0.
     *
     * For more detail, see page 667 of the Data Sheet.
     */

    if ( port < BSP_NR_GPIO_PORTS && pin < PINS_PER_PORT )
   {
       HWREG_CLEAR_SINGLE_BIT( pReg[port]->GPIO_IM, pin );
   }
}


/**
 * Clears the interrupt flag for the selected pin on the
 * selected GPIO port.
 *
 * If interrupt triggering for the pin is configured as
 * level detect, this function has no effect at all.
 *
 * Nothing is done if 'port' is greater than 5 or
 * 'pin' is greater than 7.
 *
 * @param port - desired GPIO port number (between 0 and 5)
 * @param pin - pin on the selected GPIO port (between 0 and 7)
 */
void gpio_clearInterrupt(uint8_t port, uint8_t pin)
{
    /*
     * Interrupt for a pin is cleared by setting
     * corresponding pins of the GPOICR register to 1.
     *
     * For more details, see page 670 of the Data Sheet.
     */

    if ( port < BSP_NR_GPIO_PORTS && pin < PINS_PER_PORT )
    {
        HWREG_SET_SINGLE_BIT( pReg[port]->GPIO_ICR, pin);
    }

}


/**
 * Specifies events when the selected pin triggers interrupt
 * requests (provided that interrupt requests from that
 * pin have not been masked out).
 *
 * The following options are possible:
 * - interrupt triggered on high level (signal equals 1)
 * - interrupt triggered on low level (signal equals 0)
 * - interrupt triggered on rising edge (transition of signal from 0 to 1)
 * - interrupt triggered on falling edge (transition of signal from 1 to 0)
 * - interrupt triggered on any edge (any transition of signal from 0 to 1 or vice versa)
 *
 * Nothing is done if 'port' is greater than 5 or
 * 'pin' is greater than 7 or 'mode' is invalid
 *
 * @param port - desired GPIO port number (between 0 and 5)
 * @param pin - pin on the selected GPIO port (between 0 and 7)
 * @param mode - interrupt triggering mode for the 'pin' (any valid value of gpioIntType)
 */
void gpio_configInterrupt(uint8_t port, uint8_t pin, gpioIntType mode)
{
    /*
     * Interrupt triggering event are configured via GPIOIS, GPIOIBE and
     * GPIOIEV registers.
     *
     * For level-sensitive triggering of interrupts, corresponding bits
     * of the GPIOIS must be set to 1. Additionally the corresponding
     * bits of the GPIOIEV must be set appropriately, i.e. to 1 for high
     * level (1) or to 0 for low level (0). The GPIOIBE register is
     * ignored in this case.
     *
     * For edge-sensitive triggering of interrupts, corresponding bits
     * of the GPIOIS must be cleared to 0. If triggering on any edge is
     * desired, the corresponding bits of the GPIOIBE must be set and the
     * GPIOIEV is ignored. Otherwise, the corresponding bits of the GPIOIBE
     * must be cleared and the edge is determined by the corresponding bits
     * of the GPIOIEV (0 for falling edge, 1 for rising edge).
     *
     * For more details, see pp. 654 - 655 of the Data sheet.
     *
     * For more details about registers GPIOIS, GPIOIBE and GPIOIEV,
     * see pages 664, 665 and 666 of the Data Sheet, respectively.
     */

    uint32_t maskState;

    /* sanity check */
    if ( port >= BSP_NR_GPIO_PORTS || pin >= PINS_PER_PORT)
    {
        return;
    }

    /* Store the current state of GPIOIM for the desired pin: */
    maskState = HWREG_READ_SINGLE_BIT( pReg[port]->GPIO_IM, pin );

    /*
     * To prevent any false interrupts during configuration, the
     * GPIOIM for the pin will be masked out. Its state will be
     * restored later.
     */
    HWREG_CLEAR_SINGLE_BIT( pReg[port]->GPIO_IM, pin );

    /*
     * To prevent any ambiguity, clear the GPIOIBE bit to
     * allow single edge sensitive triggering.
     * If necessary, the GPIOIBE bit will be set later.
     */
    HWREG_CLEAR_SINGLE_BIT( pReg[port]->GPIO_IBE, pin );


    /*
     * Configure interrupt triggering for the selected pin
     * as described in detail above.
     */
    switch (mode)
    {
    case GPIOINT_LEVEL_HIGH:
        HWREG_SET_SINGLE_BIT( pReg[port]->GPIO_IS, pin );
        HWREG_SET_SINGLE_BIT( pReg[port]->GPIO_IEV, pin );
        break;

    case GPIOINT_LEVEL_LOW:
        HWREG_SET_SINGLE_BIT( pReg[port]->GPIO_IS, pin );
        HWREG_CLEAR_SINGLE_BIT( pReg[port]->GPIO_IEV, pin );
        break;

    case GPIOINT_EDGE_RISING:
        HWREG_CLEAR_SINGLE_BIT( pReg[port]->GPIO_IS, pin );
        HWREG_SET_SINGLE_BIT( pReg[port]->GPIO_IEV, pin );
        break;

    case GPIOINT_EDGE_FALLING:
        HWREG_CLEAR_SINGLE_BIT( pReg[port]->GPIO_IS, pin );
        HWREG_CLEAR_SINGLE_BIT( pReg[port]->GPIO_IEV, pin );
        break;

    case GPIOINT_EDGE_ANY:
        HWREG_CLEAR_SINGLE_BIT( pReg[port]->GPIO_IS, pin );
        HWREG_SET_SINGLE_BIT( pReg[port]->GPIO_IBE, pin );
        break;
    }


    /* Finally restore the original GPIOIM state */
    HWREG_SET_BITS( pReg[port]->GPIO_IM, maskState );
}


/**
 * Registers a function that handles interrupt requests
 * triggered by the specified pin.
 *
 * Nothing is done if 'port' is greater than 5
 * or 'pin' is greater than 7.
 * If 'isr' equals NULL, an internal default dummy
 * function will be "registered" instead.
 *
 * @param port - desired GPIO port number (between 0 and 5)
 * @param pin - pin on the selected GPIO port (between 0 and 7)
 * @param isr - address of the interrupt handling routine
 */
void gpio_registerIntHandler(uint8_t port, uint8_t pin, GpioPortIntHandler_t isr)
{
    if ( port < BSP_NR_GPIO_PORTS && pin < PINS_PER_PORT )
    {
        __intrIsr[port][pin] =
            (NULL!=isr ? isr : &__defaultIntHandler );
    }
}


/**
 * Unregisters the interrupt servicing routine for the specified
 * GPIO pin (actually it is replaced by an internal dummy function
 * that does not do anything).
 *
 * Nothing is done if 'port' is greater than 5
 * or 'pin' is greater than 7.
 *
 * @param port - desired GPIO port number (between 0 and 5)
 * @param pin - pin on the selected GPIO port (between 0 and 7)
 */
void gpio_unregisterIntHandler(uint8_t port, uint8_t pin)
{
    if ( port < BSP_NR_GPIO_PORTS && pin < PINS_PER_PORT )
    {
        __intrIsr[port][pin] = &__defaultIntHandler;
    }
}


/*
 * An "unofficial" function (i.e. it should only be called from
 * handlers.c and is therefore not publicly exposed in gpio.h)
 * that checks which pin on the selected GPIO port has triggered
 * an interrupt request, and runs its appropriate interrupt
 * handling routine as specified in __intrIsr.
 *
 * @param port - desired GPIO port number (between 0 and 5)
 */
void _gpio_intHandler(uint8_t port)
{
    uint8_t pin;

    /* sanity check */
    if ( port >= BSP_NR_GPIO_PORTS )
    {
        return;
    }


    /*
     * Poll all GPIOMIS bits to find the pin(s)
     * that triggered the interrupt
     */
    for ( pin=0; pin<PINS_PER_PORT; ++pin )
    {
        if ( HWREG_READ_SINGLE_BIT( pReg[port]->GPIO_MIS, pin) )
        {
            /*
             * Although this is not supposed to happen,
             * recheck ISR's address for NULL, just in case.
             */
            if ( NULL != __intrIsr[port][pin] )
            {
                ( *__intrIsr[port][pin] )();
            }

            /* Acknowledge the interrupt request source: */
            gpio_clearInterrupt(port, pin);
        }
    }
}


/**
 * Configures the selected pin as a regular digital
 * GPIO input or output. Interrupts triggered by the
 * pin, are disabled.
 *
 * Nothing is done if 'port' is greater than 5 or
 * 'pin' is greater than 7.
 *
 * @param port - desired GPIO port number (between 0 and 5)
 * @param pin - pin number (between 0 and 7)
 * @param output - 0 to set an input pin, anything else to set as an output pin
 */
void gpio_configGpioPin(uint8_t port, uint8_t pin, uint8_t output)
{

    if ( port >= BSP_NR_GPIO_PORTS || pin >= PINS_PER_PORT)
    {
        return;
    }

    /* Activate the port: */
    sysctl_enableGpioPort(port);

    /* Unlock the port: */
    pReg[port]->GPIO_LOCK = PORT_UNLOCK;

    /* and enable setting of the pin: */
    HWREG_SET_SINGLE_BIT( pReg[port]->GPIO_CR, pin );

    /* Disable analog mode: */
    gpio_disableAnalog(port, pin);

    /* Disable triggering of interrupts */
    gpio_maskInterrupt(port, pin);

    /* Set direction: */
    if ( 0 != output )
    {
        gpio_setPinAsOutput(port, pin);
    }
    else
    {
        gpio_setPinAsInput(port, pin);
    }

    /* Finally disable alternate functions and enable digital mode: */
    gpio_setAltFunction(port, pin, 0);
    gpio_enableDigital(port, pin);

}


/*
 * An "unofficial" function (i.e. it should only be called from
 * handlers.c and is therefore not publicly exposed in gpio.h)
 * that initializes the whole __intrIsr table to the default
 * dummy "handler" __defaultIntHandler.
 *
 * This function should only be called at he start of
 * an application.
 */
void _gpio_initIntHandlers(void)
{
    uint8_t port;
    uint8_t pin;

    for ( port=0; port<BSP_NR_GPIO_PORTS; ++port )
    {
        for ( pin=0; pin<PINS_PER_PORT; ++pin )
        {
            gpio_unregisterIntHandler(port, pin);
        }
    }
}
