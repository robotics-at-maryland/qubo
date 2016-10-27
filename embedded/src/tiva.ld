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
 * A linker script that defines the device's memory layout and determines
 * where each application's section is placed.
 *
 * To verify whether objects (symbols) have been placed to expected locations, run
 * 'nm' ('arm-none-eabi-nm') on the resulting ELF image.
 */


/*
 * Memory map origin and size of the built-in memory.
 */
MEMORY
{
    FLASH (rx) : ORIGIN = 0x00000000, LENGTH = 0x00040000
    SRAM (rwx) : ORIGIN = 0x20000000, LENGTH = 0x00008000
}


/*
 * This part of the linker script defines where each application's
 * section will be placed. .text (the actual code) and .rodata 
 * (constant global variables) remain unmodified during an application
 * run, so they will be placed into flash. On the other hand, .data
 * (initialized non-constant global variables) and .bss (non-initialized
 * non-constant global variables) will be placed into SRAM.
 *
 * Note> stack space has been allocated as an array in startup.c, so the
 * stack pointer is defined there.
 */
SECTIONS
{
    .text :
    {
        _text = .;
        
        /*
         * The vector table must be placed to the top of the
         * memory map. To achieve this, it was assigned to a 
         * special section called ".isr_vector"
         */
        KEEP(*(.isr_vector))
        
        /* followed by .text and .rodata: */
        *(.text*)
        *(.rodata*)
        
        _etext = .;
    } > FLASH

    /* Just to make sure that the contents does not exceed the flash size */
    . = ORIGIN(FLASH) + LENGTH(FLASH);
    
    
    /*
     * .data and .bss are placed into SRAM:
     */
    .data : AT(ADDR(.text) + SIZEOF(.text))
    {
        _data = .;
        *(.data*)
        _edata = .;
    } > SRAM

    .bss :
    {
        /* _bss and _ebss will be required during initialization */
        _bss = .;
        *(.bss*)
        _ebss = .;
    } > SRAM
    
    /* Just to make sure that the contents does not exceed the SRAM size */
    . = ORIGIN(SRAM) + LENGTH(SRAM);
}
