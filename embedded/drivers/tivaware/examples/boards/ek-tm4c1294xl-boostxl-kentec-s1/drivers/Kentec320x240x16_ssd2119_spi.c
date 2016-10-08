//*****************************************************************************
//
// Kentec320x240x16_ssd2119_spi.c - Display driver for the Kentec
//                                  BOOSTXL-K350QVG-S1 TFT display with an
//                                  SSD2119 controller and SPI interface.
//
// Copyright (c) 2016 Texas Instruments Incorporated.  All rights reserved.
// Software License Agreement
// 
// Texas Instruments (TI) is supplying this software for use solely and
// exclusively on TI's microcontroller products. The software is owned by
// TI and/or its suppliers, and is protected under applicable copyright
// laws. You may not combine this software with "viral" open-source
// software in order to form a larger program.
// 
// THIS SOFTWARE IS PROVIDED "AS IS" AND WITH ALL FAULTS.
// NO WARRANTIES, WHETHER EXPRESS, IMPLIED OR STATUTORY, INCLUDING, BUT
// NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE APPLY TO THIS SOFTWARE. TI SHALL NOT, UNDER ANY
// CIRCUMSTANCES, BE LIABLE FOR SPECIAL, INCIDENTAL, OR CONSEQUENTIAL
// DAMAGES, FOR ANY REASON WHATSOEVER.
// 
// This is part of revision 2.1.3.156 of the EK-TM4C1294XL Firmware Package.
//
//*****************************************************************************

#include <stdint.h>
#include <stdbool.h>
#include "inc/hw_gpio.h"
#include "inc/hw_ints.h"
#include "inc/hw_memmap.h"
#include "inc/hw_types.h"
#include "driverlib/ssi.h"
#include "driverlib/gpio.h"
#include "driverlib/interrupt.h"
#include "driverlib/sysctl.h"
#include "driverlib/timer.h"
#include "driverlib/rom.h"
#include "driverlib/pin_map.h"
#include "grlib/grlib.h"
#include "drivers/Kentec320x240x16_ssd2119_spi.h"

//*****************************************************************************
//
//! \addtogroup kentec320x240x16_ssd2119_spi
//! @{
//
//*****************************************************************************

//*****************************************************************************
//
// This driver operates in four different screen orientations.  They are:
//
// * Portrait - The screen is taller than it is wide, and the flex connector is
//              on the left of the display.  This is selected by defining
//              PORTRAIT.
//
// * Landscape - The screen is wider than it is tall, and the flex connector is
//               on the bottom of the display.  This is selected by defining
//               LANDSCAPE.
//
// * Portrait flip - The screen is taller than it is wide, and the flex
//                   connector is on the right of the display.  This is
//                   selected by defining PORTRAIT_FLIP.
//
// * Landscape flip - The screen is wider than it is tall, and the flex
//                    connector is on the top of the display.  This is
//                    selected by defining LANDSCAPE_FLIP.
//
// These can also be imagined in terms of screen rotation; if portrait mode is
// 0 degrees of screen rotation, landscape is 90 degrees of counter-clockwise
// rotation, portrait flip is 180 degrees of rotation, and landscape flip is
// 270 degress of counter-clockwise rotation.
//
// If no screen orientation is selected, "landscape flip" mode will be used.
//
//*****************************************************************************
#if ! defined(PORTRAIT) && ! defined(PORTRAIT_FLIP) && \
    ! defined(LANDSCAPE) && ! defined(LANDSCAPE_FLIP)
#define LANDSCAPE
#endif

//*****************************************************************************
//
// Various definitions controlling coordinate space mapping and drawing
// direction in the four supported orientations.
//
//*****************************************************************************
#ifdef PORTRAIT
#define HORIZ_DIRECTION 0x28
#define VERT_DIRECTION 0x20
#define MAPPED_X(x, y) (319 - (y))
#define MAPPED_Y(x, y) (x)
#endif
#ifdef LANDSCAPE
#define HORIZ_DIRECTION 0x00
#define VERT_DIRECTION  0x08
#define MAPPED_X(x, y) (319 - (x))
#define MAPPED_Y(x, y) (239 - (y))
#endif
#ifdef PORTRAIT_FLIP
#define HORIZ_DIRECTION 0x18
#define VERT_DIRECTION 0x10
#define MAPPED_X(x, y) (y)
#define MAPPED_Y(x, y) (239 - (x))
#endif
#ifdef LANDSCAPE_FLIP
#define HORIZ_DIRECTION 0x30
#define VERT_DIRECTION  0x38
#define MAPPED_X(x, y) (x)
#define MAPPED_Y(x, y) (y)
#endif

//*****************************************************************************
//
// Various internal SSD2119 registers name labels
//
//*****************************************************************************
#define SSD2119_DEVICE_CODE_READ_REG  0x00
#define SSD2119_OSC_START_REG         0x00
#define SSD2119_OUTPUT_CTRL_REG       0x01
#define SSD2119_LCD_DRIVE_AC_CTRL_REG 0x02
#define SSD2119_PWR_CTRL_1_REG        0x03
#define SSD2119_DISPLAY_CTRL_REG      0x07
#define SSD2119_FRAME_CYCLE_CTRL_REG  0x0b
#define SSD2119_PWR_CTRL_2_REG        0x0c
#define SSD2119_PWR_CTRL_3_REG        0x0d
#define SSD2119_PWR_CTRL_4_REG        0x0e
#define SSD2119_GATE_SCAN_START_REG   0x0f
#define SSD2119_SLEEP_MODE_1_REG      0x10
#define SSD2119_ENTRY_MODE_REG        0x11
#define SSD2119_SLEEP_MODE_2_REG      0x12
#define SSD2119_GEN_IF_CTRL_REG       0x15
#define SSD2119_PWR_CTRL_5_REG        0x1e
#define SSD2119_RAM_DATA_REG          0x22
#define SSD2119_FRAME_FREQ_REG        0x25
#define SSD2119_ANALOG_SET_REG        0x26
#define SSD2119_VCOM_OTP_1_REG        0x28
#define SSD2119_VCOM_OTP_2_REG        0x29
#define SSD2119_GAMMA_CTRL_1_REG      0x30
#define SSD2119_GAMMA_CTRL_2_REG      0x31
#define SSD2119_GAMMA_CTRL_3_REG      0x32
#define SSD2119_GAMMA_CTRL_4_REG      0x33
#define SSD2119_GAMMA_CTRL_5_REG      0x34
#define SSD2119_GAMMA_CTRL_6_REG      0x35
#define SSD2119_GAMMA_CTRL_7_REG      0x36
#define SSD2119_GAMMA_CTRL_8_REG      0x37
#define SSD2119_GAMMA_CTRL_9_REG      0x3a
#define SSD2119_GAMMA_CTRL_10_REG     0x3b
#define SSD2119_V_RAM_POS_REG         0x44
#define SSD2119_H_RAM_START_REG       0x45
#define SSD2119_H_RAM_END_REG         0x46
#define SSD2119_X_RAM_ADDR_REG        0x4e
#define SSD2119_Y_RAM_ADDR_REG        0x4f

#define ENTRY_MODE_DEFAULT      0x6830
#define MAKE_ENTRY_MODE(x)      ((ENTRY_MODE_DEFAULT & 0xff00) | (x))

//*****************************************************************************
//
// Defines for the pins that are used to communicate with the SSD2119.
//
//*****************************************************************************
#define LCD_CS_PERIPH           SYSCTL_PERIPH_GPIOP
#define LCD_CS_BASE             GPIO_PORTP_BASE
#define LCD_CS_PIN              GPIO_PIN_3

#define LCD_DC_PERIPH           SYSCTL_PERIPH_GPIOP
#define LCD_DC_BASE             GPIO_PORTP_BASE
#define LCD_DC_PIN              GPIO_PIN_4

#define LCD_RST_PERIPH          SYSCTL_PERIPH_GPIOK
#define LCD_RST_BASE            GPIO_PORTK_BASE
#define LCD_RST_PIN             GPIO_PIN_6

#define LCD_LED_PERIPH          SYSCTL_PERIPH_GPIOG
#define LCD_LED_BASE            GPIO_PORTG_BASE
#define LCD_LED_PIN             GPIO_PIN_1

//*****************************************************************************
//
// Defines for the SSI controller and pins that are used to communicate with
// the SSD2119.
//
//*****************************************************************************
#define LCD_SSI_PERIPH          SYSCTL_PERIPH_SSI3
#define LCD_SSI_BASE            SSI3_BASE

#define LCD_SSI_GPIO_PERIPH     SYSCTL_PERIPH_GPIOQ
#define LCD_SSI_GPIO_BASE       GPIO_PORTQ_BASE
#define LCD_SSI_CLK_CFG         GPIO_PQ0_SSI3CLK
#define LCD_SSI_TX_CFG          GPIO_PQ2_SSI3XDAT0
#define LCD_SSI_CLK_PIN         GPIO_PIN_0
#define LCD_SSI_TX_PIN          GPIO_PIN_2

//*****************************************************************************
//
// The dimensions of the LCD panel.
//
//*****************************************************************************
#define LCD_VERTICAL_MAX 240
#define LCD_HORIZONTAL_MAX 320

//*****************************************************************************
//
// Translates a 24-bit RGB color to a display driver-specific color.
//
// \param c is the 24-bit RGB color.  The least-significant byte is the blue
// channel, the next byte is the green channel, and the third byte is the red
// channel.
//
// This macro translates a 24-bit RGB color into a value that can be written
// into the display's frame buffer in order to reproduce that color, or the
// closest possible approximation of that color.
//
// \return Returns the display-driver specific color.
//
//*****************************************************************************
#define DPYCOLORTRANSLATE(c)    ((((c) & 0x00f80000) >> 8) |               \
                                 (((c) & 0x0000fc00) >> 5) |               \
                                 (((c) & 0x000000f8) >> 3))

//*****************************************************************************
//
// Switches Backlight ON for the LCD Panel
//
//*****************************************************************************
static inline void
LED_ON(void)
{
    GPIOPinWrite(LCD_LED_BASE, LCD_LED_PIN, LCD_LED_PIN);
}

//*****************************************************************************
//
// Switches Backlight OFF for the LCD Panel
//
//*****************************************************************************
static inline void
LED_OFF(void)
{
    GPIOPinWrite(LCD_LED_BASE, LCD_LED_PIN, 0);
}

//*****************************************************************************
//
// Writes a data word to the SSD2119.
//
//*****************************************************************************
static inline void
WriteDataSPI(uint16_t ui16Data)
{
    uint16_t pui16Data[2];

    //
    // Write the most significant byte of the data to the bus.
    //
    pui16Data[0] = (ui16Data >> 8);
    
    //
    // Write the least significant byte of the data to the bus.
    //
    pui16Data[1] = ui16Data;

    GPIOPinWrite(LCD_DC_BASE, LCD_DC_PIN, LCD_DC_PIN);
    GPIOPinWrite(LCD_CS_BASE, LCD_CS_PIN, 0);

    SSIDataPut(LCD_SSI_BASE, pui16Data[0]);
    SSIDataPut(LCD_SSI_BASE, pui16Data[1]);

    //
    // Wait until SSI0 is done transferring all the data in the transmit FIFO.
    //
    while(SSIBusy(LCD_SSI_BASE)){ }

    GPIOPinWrite(LCD_CS_BASE, LCD_CS_PIN, LCD_CS_PIN);
}

//*****************************************************************************
//
// Writes a command to the SSD2119.
//
//*****************************************************************************
static inline void
WriteCommandSPI(uint16_t ui16Data)
{
    uint16_t pui16Data[2];

    //
    // Write the most significant byte of the data to the bus.
    //
    pui16Data[0] = 0;
    
    //
    // Write the least significant byte of the data to the bus.
    //
    pui16Data[1] = ui16Data;

    GPIOPinWrite(LCD_DC_BASE, LCD_DC_PIN, 0);
    GPIOPinWrite(LCD_CS_BASE, LCD_CS_PIN, 0);

    SSIDataPut(LCD_SSI_BASE, pui16Data[0]);
    SSIDataPut(LCD_SSI_BASE, pui16Data[1]);
    
    //
    // Wait until SSI0 is done transferring all the data in the transmit FIFO.
    //
    while(SSIBusy(LCD_SSI_BASE)){ }

    GPIOPinWrite(LCD_CS_BASE, LCD_CS_PIN, LCD_CS_PIN);
}

//*****************************************************************************
//
// Initializes the pins required for the GPIO-based LCD interface.
//
// This function configures the GPIO pins used to control the LCD display
// when the basic GPIO interface is in use.  On exit, the LCD controller
// has been reset and is ready to receive command and data writes.
//
// \return None.
//
//*****************************************************************************
static void
InitSPILCDInterface(uint32_t ui32SysClock)
{
    uint32_t pui32DataRx[3];

    //
    // The SSI3 peripheral must be enabled for use.
    //
    SysCtlPeripheralEnable(LCD_SSI_PERIPH);

    //
    // For this example SSI3 is used with PortQ[3:0].  The actual port and pins
    // used may be different on your part, consult the data sheet for more
    // information.  GPIO port Q needs to be enabled so these pins can be used.
    // TODO: change this to whichever GPIO port you are using.
    //
    SysCtlPeripheralEnable(LCD_SSI_GPIO_PERIPH);

    //
    // Enable the PortP for LCD_SCS, LCD_SDC
    // Enable the PortK for LCD_RST
    // Enable the PortG for LED backlight
    //
    SysCtlPeripheralEnable(LCD_RST_PERIPH);
    SysCtlPeripheralEnable(LCD_LED_PERIPH);
    SysCtlPeripheralEnable(LCD_CS_PERIPH);

    //
    // Change PP3 into GPIO output for LCD_SCS
    // Change PP4 into GPIO output for LCD_SDC
    // Change PK6 into GPIO output for LCD_RST
    // Change PG1 into GPIO output for LED backlight
    //
    GPIOPinTypeGPIOOutput(LCD_RST_BASE, LCD_RST_PIN);
    GPIOPinTypeGPIOOutput(LCD_CS_BASE, LCD_CS_PIN);
    GPIOPinTypeGPIOOutput(LCD_DC_BASE, LCD_DC_PIN);
    GPIOPinTypeGPIOOutput(LCD_LED_BASE, LCD_LED_PIN);

    //
    // Configure the pin muxing for SSI3 functions on port Q0, and Q2.
    // This step is not necessary if your part does not support pin muxing.
    // TODO: change this to select the port/pin you are using.
    //
    GPIOPinConfigure(LCD_SSI_CLK_CFG);
    GPIOPinConfigure(LCD_SSI_TX_CFG);

    //
    // Configure the GPIO settings for the SSI pins.  This function also gives
    // control of these pins to the SSI hardware.  Consult the data sheet to
    // see which functions are allocated per pin.
    // The pins are assigned as follows:
    //      PQ0 - SSI3CLK
    //      PQ2 - SSI2TX
    // TODO: change this to select the port/pin you are using.
    //
    GPIOPinTypeSSI(LCD_SSI_GPIO_BASE, LCD_SSI_CLK_PIN | LCD_SSI_TX_PIN);

    //
    // Configure and enable the SSI port for SPI master mode.  Use SSI3,
    // system clock supply, idle clock level low and active low clock in
    // freescale SPI mode, master mode, 15 MHz SSI frequency, and 8-bit data.
    // For SPI mode, you can set the polarity of the SSI clock when the SSI
    // unit is idle.  You can also configure what clock edge you want to
    // capture data on.  Please reference the datasheet for more information on
    // the different SPI modes.
    //
    SSIConfigSetExpClk(LCD_SSI_BASE, ui32SysClock, SSI_FRF_MOTO_MODE_0,
            SSI_MODE_MASTER, 15000000, 8);

    //
    // Enable the SSI3 module.
    //
    SSIEnable(LCD_SSI_BASE);

    //
    // Read any residual data from the SSI port.  This makes sure the receive
    // FIFOs are empty, so we don't read any unwanted junk.  This is done here
    // because the SPI SSI mode is full-duplex, which allows you to send and
    // receive at the same time.  The SSIDataGetNonBlocking function returns
    // "true" when data was returned, and "false" when no data was returned.
    // The "non-blocking" function checks if there is any data in the receive
    // FIFO and does not "hang" if there isn't.
    //
    while(SSIDataGetNonBlocking(LCD_SSI_BASE, &pui32DataRx[0]))
    {
    }

}

//*****************************************************************************
//
//! Initializes the display driver.
//!
//! \param ui32SysClock is the frequency of the system clock.
//!
//! This function initializes the LCD controller and the SSD2119 display
//! controller on the panel, preparing it to display data.
//!
//! \return None.
//
//*****************************************************************************
void
Kentec320x240x16_SSD2119Init(uint32_t ui32SysClock)
{
    uint32_t ui32ClockMS, ui32Count;

    //
    // Divide by 3 to get the number of SysCtlDelay loops in 1mS.
    //
    ui32ClockMS = ui32SysClock / (3 * 1000);

    //
    // Initializes the SPI Controller for the LCD controller
    //
    InitSPILCDInterface(ui32SysClock);

    //
    // Switch off the LED backlight
    //
    LED_OFF();

    //
    // Reset the LCD
    //
    GPIOPinWrite(LCD_RST_BASE, LCD_RST_PIN, 0);
    SysCtlDelay(10 * ui32ClockMS);
    GPIOPinWrite(LCD_RST_BASE, LCD_RST_PIN, LCD_RST_PIN);
    SysCtlDelay(20 * ui32ClockMS);

    //
    // Enter sleep mode (if we are not already there).
    //
    WriteCommandSPI(SSD2119_SLEEP_MODE_1_REG);
    WriteDataSPI(0x0001);

    //
    // Set initial power parameters.
    //
    WriteCommandSPI(SSD2119_PWR_CTRL_5_REG);
    WriteDataSPI(0x00BA);
    WriteCommandSPI(SSD2119_VCOM_OTP_1_REG);
    WriteDataSPI(0x0006);

    //
    // Start the oscillator.
    //
    WriteCommandSPI(SSD2119_OSC_START_REG);
    WriteDataSPI(0x0001);

    //
    // Set pixel format and basic display orientation (scanning direction).
    //
    WriteCommandSPI(SSD2119_OUTPUT_CTRL_REG);
    WriteDataSPI(0x30EF);
    WriteCommandSPI(SSD2119_LCD_DRIVE_AC_CTRL_REG);
    WriteDataSPI(0x0600);

    //
    // Exit sleep mode.
    //
    WriteCommandSPI(SSD2119_SLEEP_MODE_1_REG);
    WriteDataSPI(0x0000);

    //
    // Delay 30mS
    //
    SysCtlDelay(30 * ui32ClockMS);

    //
    // Configure pixel color format and MCU interface parameters.
    //
    WriteCommandSPI(SSD2119_ENTRY_MODE_REG);
    WriteDataSPI(ENTRY_MODE_DEFAULT);

    //
    // Enable the display.
    //
    WriteCommandSPI(SSD2119_DISPLAY_CTRL_REG);
    WriteDataSPI(0x0033);

    //
    // Set VCIX2 voltage to 6.1V.
    //
    WriteCommandSPI(SSD2119_PWR_CTRL_2_REG);
    WriteDataSPI(0x0005);

    //
    // Configure gamma correction.
    //
    WriteCommandSPI(SSD2119_GAMMA_CTRL_1_REG);
    WriteDataSPI(0x0000);
    WriteCommandSPI(SSD2119_GAMMA_CTRL_2_REG);
    WriteDataSPI(0x0400);
    WriteCommandSPI(SSD2119_GAMMA_CTRL_3_REG);
    WriteDataSPI(0x0106);
    WriteCommandSPI(SSD2119_GAMMA_CTRL_4_REG);
    WriteDataSPI(0x0700);
    WriteCommandSPI(SSD2119_GAMMA_CTRL_5_REG);
    WriteDataSPI(0x0002);
    WriteCommandSPI(SSD2119_GAMMA_CTRL_6_REG);
    WriteDataSPI(0x0702);
    WriteCommandSPI(SSD2119_GAMMA_CTRL_7_REG);
    WriteDataSPI(0x0707);
    WriteCommandSPI(SSD2119_GAMMA_CTRL_8_REG);
    WriteDataSPI(0x0203);
    WriteCommandSPI(SSD2119_GAMMA_CTRL_9_REG);
    WriteDataSPI(0x1400);
    WriteCommandSPI(SSD2119_GAMMA_CTRL_10_REG);
    WriteDataSPI(0x0F03);

    //
    // Configure Vlcd63 and VCOMl.
    //
    WriteCommandSPI(SSD2119_PWR_CTRL_3_REG);
    WriteDataSPI(0x0007);
    WriteCommandSPI(SSD2119_PWR_CTRL_4_REG);
    WriteDataSPI(0x3100);

    //
    // Set the display size and ensure that the GRAM window is set to allow
    // access to the full display buffer.
    //
    WriteCommandSPI(SSD2119_V_RAM_POS_REG);
    WriteDataSPI((LCD_VERTICAL_MAX-1) << 8);
    WriteCommandSPI(SSD2119_H_RAM_START_REG);
    WriteDataSPI(0x0000);
    WriteCommandSPI(SSD2119_H_RAM_END_REG);
    WriteDataSPI(LCD_HORIZONTAL_MAX-1);
    WriteCommandSPI(SSD2119_X_RAM_ADDR_REG);
    WriteDataSPI(0x00);
    WriteCommandSPI(SSD2119_Y_RAM_ADDR_REG);
    WriteDataSPI(0x00);

    //
    // Clear the contents of the display buffer.
    //
    WriteCommandSPI(SSD2119_RAM_DATA_REG);
    for(ui32Count = 0; ui32Count < (320 * 240); ui32Count++)
    {
        WriteDataSPI(0x0000);
    }

    //
    // Switch on the LED backlight
    //
    LED_ON();
}

//*****************************************************************************
//
//! Draws a pixel on the screen.
//!
//! \param pvDisplayData is a pointer to the driver-specific data for this
//! display driver.
//! \param i32X is the X coordinate of the pixel.
//! \param i32Y is the Y coordinate of the pixel.
//! \param ui32Value is the color of the pixel.
//!
//! This function sets the given pixel to a particular color.  The coordinates
//! of the pixel are assumed to be within the extents of the display.
//!
//! \return None.
//
//*****************************************************************************
static void
Kentec320x240x16_SSD2119PixelDraw(void *pvDisplayData, int32_t i32X,
        int32_t i32Y,
        uint32_t ui32Value)
{
    //
    // Set the X address of the display cursor.
    //
    WriteCommandSPI(SSD2119_X_RAM_ADDR_REG);
    WriteDataSPI(MAPPED_X(i32X, i32Y));

    //
    // Set the Y address of the display cursor.
    //
    WriteCommandSPI(SSD2119_Y_RAM_ADDR_REG);
    WriteDataSPI(MAPPED_Y(i32X, i32Y));

    //
    // Write the pixel value.
    //
    WriteCommandSPI(SSD2119_RAM_DATA_REG);
    WriteDataSPI(ui32Value);
}

//*****************************************************************************
//
//! Draws a horizontal sequence of pixels on the screen.
//!
//! \param pvDisplayData is a pointer to the driver-specific data for this
//! display driver.
//! \param i32X is the X coordinate of the first pixel.
//! \param i32Y is the Y coordinate of the first pixel.
//! \param i32X0 is sub-pixel offset within the pixel data, which is valid for
//! 1 or 4 bit per pixel formats.
//! \param i32Count is the number of pixels to draw.
//! \param i32BPP is the number of bits per pixel; must be 1, 4, or 8.
//! \param pui8Data is a pointer to the pixel data.  For 1 and 4 bit per pixel
//! formats, the most significant bit(s) represent the left-most pixel.
//! \param pui8Palette is a pointer to the palette used to draw the pixels.
//!
//! This function draws a horizontal sequence of pixels on the screen, using
//! the supplied palette.  For 1 bit per pixel format, the palette contains
//! pre-translated colors; for 4 and 8 bit per pixel formats, the palette
//! contains 24-bit RGB values that must be translated before being written to
//! the display.
//!
//! \return None.
//
//*****************************************************************************
static void
Kentec320x240x16_SSD2119PixelDrawMultiple(void *pvDisplayData, int32_t i32X,
                                           int32_t i32Y, int32_t i32X0,
                                           int32_t i32Count, int32_t i32BPP,
                                           const uint8_t *pui8Data,
                                           const uint8_t *pui8Palette)
{
    uint32_t ui32Byte;

    //
    // Set the cursor increment to left to right, followed by top to bottom.
    //
    WriteCommandSPI(SSD2119_ENTRY_MODE_REG);
    WriteDataSPI(MAKE_ENTRY_MODE(HORIZ_DIRECTION));

    //
    // Set the starting X address of the display cursor.
    //
    WriteCommandSPI(SSD2119_X_RAM_ADDR_REG);
    WriteDataSPI(MAPPED_X(i32X, i32Y));

    //
    // Set the Y address of the display cursor.
    //
    WriteCommandSPI(SSD2119_Y_RAM_ADDR_REG);
    WriteDataSPI(MAPPED_Y(i32X, i32Y));

    //
    // Write the data RAM write command.
    //
    WriteCommandSPI(SSD2119_RAM_DATA_REG);

    //
    // Determine how to interpret the pixel data based on the number of bits
    // per pixel.
    //
    switch(i32BPP & ~GRLIB_DRIVER_FLAG_NEW_IMAGE)
    {
        //
        // The pixel data is in 1 bit per pixel format.
        //
        case 1:
        {
            //
            // Loop while there are more pixels to draw.
            //
            while(i32Count)
            {
                //
                // Get the next byte of image data.
                //
                ui32Byte = *pui8Data++;

                //
                // Loop through the pixels in this byte of image data.
                //
                for(; (i32X0 < 8) && i32Count; i32X0++, i32Count--)
                {
                    //
                    // Draw this pixel in the appropriate color.
                    //
                    WriteDataSPI(((uint32_t *)pui8Palette)
                            [(ui32Byte >> (7 - i32X0)) & 1]);
                }

                //
                // Start at the beginning of the next byte of image data.
                //
                i32X0 = 0;
            }

            //
            // The image data has been drawn.
            //
            break;
        }

        //
        // The pixel data is in 4 bit per pixel format.
        //
        case 4:
        {
            //
            // Loop while there are more pixels to draw.  "Duff's device" is
            // used to jump into the middle of the loop if the first nibble of
            // the pixel data should not be used.  Duff's device makes use of
            // the fact that a case statement is legal anywhere within a
            // sub-block of a switch statement.  See
            // http://en.wikipedia.org/wiki/Duff's_device for detailed
            // information about Duff's device.
            //
            switch(i32X0 & 1)
            {
                case 0:
                    while(i32Count)
                    {
                        //
                        // Get the upper nibble of the next byte of pixel data
                        // and extract the corresponding entry from the
                        // palette.
                        //
                        ui32Byte = (*pui8Data >> 4) * 3;
                        ui32Byte = (*(uint32_t *)(pui8Palette + ui32Byte) &
                                  0x00ffffff);

                        //
                        // Translate this palette entry and write it to the
                        // screen.
                        //
                        WriteDataSPI(DPYCOLORTRANSLATE(ui32Byte));

                        //
                        // Decrement the count of pixels to draw.
                        //
                        i32Count--;

                        //
                        // See if there is another pixel to draw.
                        //
                        if(i32Count)
                        {
                case 1:
                            //
                            // Get the lower nibble of the next byte of pixel
                            // data and extract the corresponding entry from
                            // the palette.
                            //
                            ui32Byte = (*pui8Data++ & 15) * 3;
                            ui32Byte = (*(uint32_t *)(pui8Palette + ui32Byte) &
                                      0x00ffffff);

                            //
                            // Translate this palette entry and write it to the
                            // screen.
                            //
                            WriteDataSPI(DPYCOLORTRANSLATE(ui32Byte));

                            //
                            // Decrement the count of pixels to draw.
                            //
                            i32Count--;
                        }
                    }
            }

            //
            // The image data has been drawn.
            //
            break;
        }

        //
        // The pixel data is in 8 bit per pixel format.
        //
        case 8:
        {
            //
            // Loop while there are more pixels to draw.
            //
            while(i32Count--)
            {
                //
                // Get the next byte of pixel data and extract the
                // corresponding entry from the palette.
                //
                ui32Byte = *pui8Data++ * 3;
                ui32Byte = *(uint32_t *)(pui8Palette + ui32Byte) & 0x00ffffff;

                //
                // Translate this palette entry and write it to the screen.
                //
                WriteDataSPI(DPYCOLORTRANSLATE(ui32Byte));
            }

            //
            // The image data has been drawn.
            //
            break;
        }

        //
        // We are being passed data in the display's native format.  Merely
        // write it directly to the display.  This is a special case which is
        // not used by the graphics library but which is helpful to
        // applications which may want to handle, for example, JPEG images.
        //
        case 16:
        {
            uint16_t ui16Byte;

            //
            // Loop while there are more pixels to draw.
            //
            while(i32Count--)
            {
                //
                // Get the next byte of pixel data and extract the
                // corresponding entry from the palette.
                //
                ui16Byte = *((uint16_t *)pui8Data);
                pui8Data += 2;

                //
                // Translate this palette entry and write it to the screen.
                //
                WriteDataSPI(ui16Byte);
            }
        }
    }
}

//*****************************************************************************
//
//! Draws a horizontal line.
//!
//! \param pvDisplayData is a pointer to the driver-specific data for this
//! display driver.
//! \param i32X1 is the X coordinate of the start of the line.
//! \param i32X2 is the X coordinate of the end of the line.
//! \param i32Y is the Y coordinate of the line.
//! \param ui32Value is the color of the line.
//!
//! This function draws a horizontal line on the display.  The coordinates of
//! the line are assumed to be within the extents of the display.
//!
//! \return None.
//
//*****************************************************************************
static void
Kentec320x240x16_SSD2119LineDrawH(void *pvDisplayData, int32_t i32X1,
        int32_t i32X2, int32_t i32Y,
        uint32_t ui32Value)
{
    //
    // Set the cursor increment to left to right, followed by top to bottom.
    //
    WriteCommandSPI(SSD2119_ENTRY_MODE_REG);
    WriteDataSPI(MAKE_ENTRY_MODE(HORIZ_DIRECTION));

    //
    // Set the starting X address of the display cursor.
    //
    WriteCommandSPI(SSD2119_X_RAM_ADDR_REG);
    WriteDataSPI(MAPPED_X(i32X1, i32Y));

    //
    // Set the Y address of the display cursor.
    //
    WriteCommandSPI(SSD2119_Y_RAM_ADDR_REG);
    WriteDataSPI(MAPPED_Y(i32X1, i32Y));

    //
    // Write the data RAM write command.
    //
    WriteCommandSPI(SSD2119_RAM_DATA_REG);

    //
    // Loop through the pixels of this horizontal line.
    //
    while(i32X1++ <= i32X2)
    {
        //
        // Write the pixel value.
        //
        WriteDataSPI(ui32Value);
    }
}

//*****************************************************************************
//
//! Draws a vertical line.
//!
//! \param pvDisplayData is a pointer to the driver-specific data for this
//! display driver.
//! \param i32X is the X coordinate of the line.
//! \param i32Y1 is the Y coordinate of the start of the line.
//! \param i32Y2 is the Y coordinate of the end of the line.
//! \param ui32Value is the color of the line.
//!
//! This function draws a vertical line on the display.  The coordinates of the
//! line are assumed to be within the extents of the display.
//!
//! \return None.
//
//*****************************************************************************
static void
Kentec320x240x16_SSD2119LineDrawV(void *pvDisplayData, int32_t i32X,
        int32_t i32Y1, int32_t i32Y2,
        uint32_t ui32Value)
{
    //
    // Set the cursor increment to top to bottom, followed by left to right.
    //
    WriteCommandSPI(SSD2119_ENTRY_MODE_REG);
    WriteDataSPI(MAKE_ENTRY_MODE(VERT_DIRECTION));

    //
    // Set the X address of the display cursor.
    //
    WriteCommandSPI(SSD2119_X_RAM_ADDR_REG);
    WriteDataSPI(MAPPED_X(i32X, i32Y1));

    //
    // Set the starting Y address of the display cursor.
    //
    WriteCommandSPI(SSD2119_Y_RAM_ADDR_REG);
    WriteDataSPI(MAPPED_Y(i32X, i32Y1));

    //
    // Write the data RAM write command.
    //
    WriteCommandSPI(SSD2119_RAM_DATA_REG);

    //
    // Loop through the pixels of this vertical line.
    //
    while(i32Y1++ <= i32Y2)
    {
        //
        // Write the pixel value.
        //
        WriteDataSPI(ui32Value);
    }
}

//*****************************************************************************
//
//! Fills a rectangle.
//!
//! \param pvDisplayData is a pointer to the driver-specific data for this
//! display driver.
//! \param pRect is a pointer to the structure describing the rectangle.
//! \param ui32Value is the color of the rectangle.
//!
//! This function fills a rectangle on the display.  The coordinates of the
//! rectangle are assumed to be within the extents of the display, and the
//! rectangle specification is fully inclusive (in other words, both sXMin and
//! sXMax are drawn, along with sYMin and sYMax).
//!
//! \return None.
//
//*****************************************************************************
static void
Kentec320x240x16_SSD2119RectFill(void *pvDisplayData, const tRectangle *pRect,
                                 uint32_t ui32Value)
{
    int32_t i32Count;

    //
    // Write the Y extents of the rectangle.
    //
    WriteCommandSPI(SSD2119_ENTRY_MODE_REG);
    WriteDataSPI(MAKE_ENTRY_MODE(HORIZ_DIRECTION));

    //
    // Write the X extents of the rectangle.
    //
    WriteCommandSPI(SSD2119_H_RAM_START_REG);
#if (defined PORTRAIT) || (defined LANDSCAPE)
    WriteDataSPI(MAPPED_X(pRect->i16XMax, pRect->i16YMax));
#else
    WriteDataSPI(MAPPED_X(pRect->i16XMin, pRect->i16YMin));
#endif

    WriteCommandSPI(SSD2119_H_RAM_END_REG);
#if (defined PORTRAIT) || (defined LANDSCAPE)
    WriteDataSPI(MAPPED_X(pRect->i16XMin, pRect->i16YMin));
#else
    WriteDataSPI(MAPPED_X(pRect->i16XMax, pRect->i16YMax));
#endif

    //
    // Write the Y extents of the rectangle
    //
    WriteCommandSPI(SSD2119_V_RAM_POS_REG);
#if (defined LANDSCAPE_FLIP) || (defined PORTRAIT)
    WriteDataSPI(MAPPED_Y(pRect->i16XMin, pRect->i16YMin) |
             (MAPPED_Y(pRect->i16XMax, pRect->i16YMax) << 8));
#else
    WriteDataSPI(MAPPED_Y(pRect->i16XMax, pRect->i16YMax) |
             (MAPPED_Y(pRect->i16XMin, pRect->i16YMin) << 8));
#endif

    //
    // Set the display cursor to the upper left of the rectangle (in
    // application coordinate space).
    //
    WriteCommandSPI(SSD2119_X_RAM_ADDR_REG);
    WriteDataSPI(MAPPED_X(pRect->i16XMin, pRect->i16YMin));

    WriteCommandSPI(SSD2119_Y_RAM_ADDR_REG);
    WriteDataSPI(MAPPED_Y(pRect->i16XMin, pRect->i16YMin));

    //
    // Tell the controller we are about to write data into its RAM.
    //
    WriteCommandSPI(SSD2119_RAM_DATA_REG);

    //
    // Loop through the pixels of this filled rectangle.
    //
    for(i32Count = ((pRect->i16XMax - pRect->i16XMin + 1) *
                  (pRect->i16YMax - pRect->i16YMin + 1));
                  i32Count >= 0; i32Count--)
    {
        //
        // Write the pixel value.
        //
        WriteDataSPI(ui32Value);
    }

    //
    // Reset the X extents to the entire screen.
    //
    WriteCommandSPI(SSD2119_H_RAM_START_REG);
    WriteDataSPI(0x0000);
    WriteCommandSPI(SSD2119_H_RAM_END_REG);
    WriteDataSPI(0x013F);

    //
    // Reset the Y extent to the full screen
    //
    WriteCommandSPI(SSD2119_V_RAM_POS_REG);
    WriteDataSPI(0xEF00);
}

//*****************************************************************************
//
//! Translates a 24-bit RGB color to a display driver-specific color.
//!
//! \param pvDisplayData is a pointer to the driver-specific data for this
//! display driver.
//! \param ui32Value is the 24-bit RGB color.  The least-significant byte is 
//! the blue channel, the next byte is the green channel, and the third byte is
//! the red channel.
//!
//! This function translates a 24-bit RGB color into a value that can be
//! written into the display's frame buffer in order to reproduce that color,
//! or the closest possible approximation of that color.
//!
//! \return Returns the display-driver specific color.
//
//*****************************************************************************
static uint32_t
Kentec320x240x16_SSD2119ColorTranslate(void *pvDisplayData,
                                       uint32_t ui32Value)
{
    //
    // Translate from a 24-bit RGB color to a 5-6-5 RGB color.
    //
    return(DPYCOLORTRANSLATE(ui32Value));
}

//*****************************************************************************
//
//! Flushes any cached drawing operations.
//!
//! \param pvDisplayData is a pointer to the driver-specific data for this
//! display driver.
//!
//! This functions flushes any cached drawing operations to the display.  This
//! is useful when a local frame buffer is used for drawing operations, and the
//! flush would copy the local frame buffer to the display.  For the SSD2119
//! driver, the flush is a no operation.
//!
//! \return None.
//
//*****************************************************************************
static void
Kentec320x240x16_SSD2119Flush(void *pvDisplayData)
{
    //
    // There is nothing to be done.
    //
}

//*****************************************************************************
//
//! The display structure that describes the driver for the Kentec
//! K350QVG-V2-F TFT panel with an SSD2119 controller.
//
//*****************************************************************************
const tDisplay g_sKentec320x240x16_SSD2119 =
{
    sizeof(tDisplay),
    0,
#if defined(PORTRAIT) || defined(PORTRAIT_FLIP)
    240,
    320,
#else
    320,
    240,
#endif
    Kentec320x240x16_SSD2119PixelDraw,
    Kentec320x240x16_SSD2119PixelDrawMultiple,
    Kentec320x240x16_SSD2119LineDrawH,
    Kentec320x240x16_SSD2119LineDrawV,
    Kentec320x240x16_SSD2119RectFill,
    Kentec320x240x16_SSD2119ColorTranslate,
    Kentec320x240x16_SSD2119Flush
};

//*****************************************************************************
//
// Close the Doxygen group.
//! @}
//
//*****************************************************************************
