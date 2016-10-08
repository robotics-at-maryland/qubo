//*****************************************************************************
//
// eeprom_pb.c - EEPROM parameter block functions.
//
// Copyright (c) 2014-2016 Texas Instruments Incorporated.  All rights reserved.
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
// This is part of revision 2.1.3.156 of the Tiva Utility Library.
//
//*****************************************************************************

#include <stdbool.h>
#include <stdint.h>
#include "inc/hw_types.h"
#include "inc/hw_sysctl.h"
#include "driverlib/debug.h"
#include "driverlib/eeprom.h"
#include "driverlib/rom.h"
#include "driverlib/sysctl.h"
#include "utils/eeprom_pb.h"

//*****************************************************************************
//
//! \addtogroup eeprom_pb_api
//! @{
//!
//! This module is based on the flash_pb module, which uses flash sectors at
//! the end of flash memory to store application parameter blocks.  This module
//! replaces the use of flash memory with the use of the EEPROM module,
//! available in some Tiva devices.
//!
//! This module does not attempt to optimize the usage of EEPROM memory, but
//! instead creates an API that will allow a very easy drop-in replacement in
//! applications that are already using the flash_pb module.
//
//*****************************************************************************

//*****************************************************************************
//
//! Default parameter block shadow size.
//
//*****************************************************************************
#ifndef EEPROM_PB_SHADOW_SIZE
#define EEPROM_PB_SHADOW_SIZE   512
#endif

//
// Sanity check on EEPROM_PB_SHADOW_SIZE
//
#if ((EEPROM_PB_SHADOW_SIZE & 3) != 0)
#error "EEPROM_PB_SHADOW_SIZE is not word-aligned!"
#endif

#if (EEPROM_PB_SHADOW_SIZE > (6 * 1024))
#warning "EEPROM_PB_SHADOW size may exceed available EEPROM memory size!"
#endif

//*****************************************************************************
//
// The address of the beginning of the eeprom used for storing parameter
// blocks; this must be an integer multiple of 4 (that is, 32-bit word
// aligned).
//
//*****************************************************************************
static uint32_t g_ui32EEPROMPBStart;

//*****************************************************************************
//
// The size of the parameter block when stored in flash; this must be an
// integer multiple of 4 (that is, integer number of 32-bit words).
//
//*****************************************************************************
static uint32_t g_ui32EEPROMPBSize;

//*****************************************************************************
//
// The address of the most recent parameter block in flash.
//
//*****************************************************************************
static uint8_t *g_pui8EEPROMPBCurrent;

//*****************************************************************************
//
// The shadow buffer for the EEPROM parameter block contents.
//
//*****************************************************************************
static uint32_t g_pui32EEPROMPBShadowBuffer[EEPROM_PB_SHADOW_SIZE];

//*****************************************************************************
//
//! Determines if the parameter block at the given address is valid.
//!
//! \param pui8Offset is the address of the parameter block to check.
//!
//! This function will compute the checksum of a parameter block in flash to
//! determine if it is valid.
//!
//! \return Returns one if the parameter block is valid and zero if it is not.
//
//*****************************************************************************
static uint32_t
EEPROMPBIsValid(uint8_t *pui8Offset)
{
    uint32_t ui32Idx, ui32Sum;

    //
    // Check the arguments.
    //
    ASSERT(pui8Offset != (void *)0);

    //
    // Loop through the bytes in the block, computing the checksum.
    //
    for(ui32Idx = 0, ui32Sum = 0; ui32Idx < g_ui32EEPROMPBSize; ui32Idx++)
    {
        ui32Sum += pui8Offset[ui32Idx];
    }

    //
    // The checksum should be zero, so return a failure if it is not.
    //
    if((ui32Sum & 255) != 0)
    {
        return(0);
    }

    //
    // If the sum is equal to the size * 255, then the block is all ones and
    // should not be considered valid.
    //
    if((g_ui32EEPROMPBSize * 255) == ui32Sum)
    {
        return(0);
    }

    //
    // This is a valid parameter block.
    //
    return(1);
}

//*****************************************************************************
//
//! Gets the address of the most recent parameter block.
//!
//! This function returns the address of the most recent parameter block that
//! is stored in eeprom.  In most cases, the current buffer will be the same
//! as the shadow buffer.
//!
//! \return Returns the address of the most recent parameter block, or NULL if
//! there is no valid parameter block in the EEPROM.
//
//*****************************************************************************
uint8_t *
EEPROMPBGet(void)
{
    //
    // See if there is a valid parameter block.
    //
    if(g_pui8EEPROMPBCurrent)
    {
        //
        // Return the address of the most recent parameter block.
        //
        return(g_pui8EEPROMPBCurrent);
    }

    //
    // There are no valid parameter blocks in flash, so return NULL.
    //
    return(0);
}

//*****************************************************************************
//
//! Writes a new parameter block to flash.
//!
//! \param pui8Buffer is the address of the parameter block to be written to
//! flash.
//!
//! This function will write a parameter block to flash.  Saving the new
//! parameter blocks involves three steps:
//!
//! - Setting the sequence number such that it is one greater than the sequence
//!   number of the latest parameter block in flash.
//! - Computing the checksum of the parameter block.
//! - Writing the parameter block into the storage immediately following the
//!   latest parameter block in flash; if that storage is at the start of an
//!   erase block, that block is erased first.
//!
//! By this process, there is always a valid parameter block in flash.  If
//! power is lost while writing a new parameter block, the checksum will not
//! match and the partially written parameter block will be ignored.  This is
//! what makes this fault-tolerant.
//!
//! Another benefit of this scheme is that it provides wear leveling on the
//! flash.  Since multiple parameter blocks fit into each erase block of flash,
//! and multiple erase blocks are used for parameter block storage, it takes
//! quite a few parameter block saves before flash is re-written.
//!
//! \return None.
//
//*****************************************************************************
void
EEPROMPBSave(uint8_t *pui8Buffer)
{
    uint8_t *pui8New;
    uint32_t ui32Idx, ui32Sum, ui32Data;

    //
    // Check the arguments.
    //
    ASSERT(pui8Buffer != (void *)0);

    //
    // See if there is a valid parameter block in flash.
    //
    if(g_pui8EEPROMPBCurrent)
    {
        //
        // Set the sequence number to one greater than the most recent
        // parameter block.
        //
        pui8Buffer[0] = g_pui8EEPROMPBCurrent[0] + 1;

    }
    else
    {
        //
        // There is not a valid parameter block in flash, so set the sequence
        // number of this parameter block to zero.
        //
        pui8Buffer[0] = 0;
    }

    //
    // Compute the checksum of the parameter block to be written.
    //
    for(ui32Idx = 0, ui32Sum = 0; ui32Idx < g_ui32EEPROMPBSize; ui32Idx++)
    {
        ui32Sum -= pui8Buffer[ui32Idx];
    }

    //
    // Store the checksum into the parameter block.
    //
    pui8Buffer[1] += ui32Sum;

    //
    // Write this parameter block to flash.
    //
    for(ui32Idx = 0; ui32Idx < g_ui32EEPROMPBSize; ui32Idx += 4)
    {
        ui32Data = *(uint32_t *)(pui8Buffer + ui32Idx);
        ui32Sum = EEPROMProgram(&ui32Data, g_ui32EEPROMPBStart + ui32Idx, 4);
        if(ui32Sum != 0)
        {
            g_pui8EEPROMPBCurrent = (uint8_t *)0;
            return;
        }
    }

    //
    // Read back what was written to EEPROM into the shadow buffer.
    //
    EEPROMRead(g_pui32EEPROMPBShadowBuffer, g_ui32EEPROMPBStart,
               g_ui32EEPROMPBSize);

    //
    // See if this is a valid parameter block (in other words, the checksum
    // is correct).
    //
    if(!EEPROMPBIsValid((uint8_t *)g_pui32EEPROMPBShadowBuffer))
    {
        //
        // Indicate that we have no valid parameter block.
        //
        g_pui8EEPROMPBCurrent = (uint8_t *)0;
        return;
    }

    //
    // Compare the parameter block data to the data that should now be in
    // flash.  Return if any of the data does not compare, leaving the previous
    // parameter block in flash as the most recent (since the current parameter
    // block failed to properly program).
    //
    pui8New = (uint8_t *)g_pui32EEPROMPBShadowBuffer;
    for(ui32Idx = 0; ui32Idx < g_ui32EEPROMPBSize; ui32Idx++)
    {
        if(pui8New[ui32Idx] != pui8Buffer[ui32Idx])
        {
            g_pui8EEPROMPBCurrent = (uint8_t *)0;
            return;
        }
    }

    //
    // The new parameter block becomes the most recent parameter block.
    //
    g_pui8EEPROMPBCurrent = pui8New;
}

//*****************************************************************************
//
//! Initializes the eeprom parameter block.
//!
//! \param ui32Start is the offset from the beginning of the EEPROM memory to
//! be used for storing the parameter block;  this must be an integer multiple
//! of 4 (that is, word aligned)
//! \param ui32Size is the size of the parameter block when stored in flash;
//! this must be an integer multiple of 4 (that is integer number of words).
//!
//! This function initialize the EEPROM module to be used for parameter block
//! storage.
//!
//! A parameter block is an array of bytes that contain the persistent
//! parameters for the application.  The only special requirement for the
//! parameter block is that the first byte is a sequence number (explained
//! in EEPROMPBSave()) and the second byte is a checksum used to validate the
//! correctness of the data (the checksum byte is the byte such that the sum of
//! all bytes in the parameter block is zero).
//!
//! This function must be called before any other eeprom parameter block
//! functions are called.
//!
//! \return 0 if successfully intialized, 1 otherwise.
//
//*****************************************************************************
uint32_t
EEPROMPBInit(uint32_t ui32Start, uint32_t ui32Size)
{
    uint32_t ui32RetValue;

    //
    // Check the arguments.
    //
    ASSERT((ui32Start & 3) == 0);
    ASSERT((ui32Size & 3) == 0);
    ASSERT(ui32Size <= EEPROM_PB_SHADOW_SIZE);

    //
    // Save the characteristics of the flash memory to be used for storing
    // parameter blocks.
    //
    g_ui32EEPROMPBStart = ui32Start;
    g_ui32EEPROMPBSize = ui32Size;

    //
    // Enable the EEPROM peripheral, with a delay to allow clocks/power
    // to stabilize.
    //
    SysCtlPeripheralEnable(SYSCTL_PERIPH_EEPROM0);

    //
    // Initialize the EEPROM
    //
    ui32RetValue = EEPROMInit();
    if(ui32RetValue != EEPROM_INIT_OK)
    {
        g_pui8EEPROMPBCurrent = (uint8_t *)0;
        return(1);
    }

    //
    // Verify that the parameter block size will fit within the EEPROM size.
    //
    ui32RetValue = EEPROMSizeGet();
    if(ui32Size > ui32RetValue)
    {
        g_pui8EEPROMPBCurrent = (uint8_t *)0;
        return(1);
    }

    //
    // Read the existing contents of the EEPROM into the shadow buffer.
    //
    EEPROMRead(g_pui32EEPROMPBShadowBuffer, g_ui32EEPROMPBStart,
               g_ui32EEPROMPBSize);

    //
    // See if this is a valid parameter block (in other words, the checksum
    // is correct).
    //
    if(EEPROMPBIsValid((uint8_t *)g_pui32EEPROMPBShadowBuffer))
    {
        //
        // Save the address of the shadow parameter block found.  If the
        // parameter block is not valid, this will be a NULL pointer.
        //
        g_pui8EEPROMPBCurrent = (uint8_t *)g_pui32EEPROMPBShadowBuffer;
    }
    else
    {
        g_pui8EEPROMPBCurrent = (uint8_t *)0;
    }

    //
    // Return a success code here.
    //
    return(0);
}

//*****************************************************************************
//
// Close the Doxygen group.
//! @}
//
//*****************************************************************************
