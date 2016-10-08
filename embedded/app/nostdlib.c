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
 * A collection of stdlib "clones", required by FreeRTOS.
 *
 * If the standard C library is going to be linked to the application,
 * do not link this file!
 *
 * @author Jernej Kovacic
 */

#include <stddef.h>
#include <stdint.h>

/* A convenience macro that defines the upper limit of 'size_t' */
#define SIZE_T_MAX     ( (size_t) (-1) )


/*
 * @param x - first value
 * @param y - second value
 *
 * @return smaller of both input values
 */
static inline size_t minval(size_t x, size_t y)
{
    return ( x<=y ? x : y );
}


/**
 * Fill block of memory.
 *
 * Sets the first 'num' bytes of the block of memory pointed by 'ptr' to the
 * specified 'value' (interpreted as an unsigned char).
 *
 * @param ptr - pointer to the block of memory to fill
 * @param value - value to be set, passed as an int and converted to unsigned char
 * @param num - number of bytes to be set to the 'value'.
 *
 * @return 'ptr' is returned
 */
void* memset(void* ptr, int value, size_t num )
{
    unsigned char* p = (unsigned char*) ptr;
    size_t n = num;

    /* sanity check */
    if ( NULL==ptr )
    {
        goto endf;
    }

    /*
     * If destination block exceeds the range of 'size_t',
     * decrease 'num' accordingly.
     */
    if ( num > (size_t) ((unsigned char*) SIZE_T_MAX - p) )
    {
        n = (unsigned char*) SIZE_T_MAX - p;
        /* TODO or maybe just goto endf???? */
    }

    /* Set 'value' to each byte of the block: */
    while (n--)
    {
        *(p++) = (unsigned char) value;
    }

endf:
    return ptr;
}


/**
 * Copy block of memory.
 *
 * Copies the values of 'num' bytes from the location pointed by 'source'
 * directly to the memory block pointed by 'destination'.
 *
 * The underlying type of the objects pointed by both the 'source' and
 * 'destination' pointers are irrelevant for this function; The result is
 * a binary copy of the data.
 *
 * The function does not check for any terminating null character in 'source' -
 * it always copies exactly 'num' bytes.
 *
 * If any block exceeds range of 'size_t', 'num' is decreased accordingly.
 *
 * The function copies the source block correctly even if both blocks overlap.
 *
 * @param destination - pointer to the destination array where the content is to be copied
 * @param source - pointer to the source of data to be copied
 * @param num - number of bytes to copy
 *
 * @return 'destination' is returned or NULL if any parameter equals NULL
 */
void* memcpy(void* destination, const void* source, size_t num )
{
    unsigned char* srcptr = (unsigned char*) source;
    unsigned char* destptr = (unsigned char*) destination;
    size_t n = num;

    /* sanity check */
    if ( NULL==source || NULL==destination )
    {
        return NULL;
    }

    /* Nothing to do if attempting to copy to itself: */
    if ( srcptr == destptr )
    {
        return destination;
    }

    /*
     * If any block exceeds the range of 'size_t',
     * decrease 'num' accordingly.
     */
    if ( num > (size_t) ((unsigned char*) SIZE_T_MAX-destptr) ||
         num > (size_t) ((unsigned char*) SIZE_T_MAX-srcptr) )
    {
        n = minval((unsigned char*) SIZE_T_MAX-destptr,
                   (unsigned char*) SIZE_T_MAX-srcptr);
        /* TODO or maybe just return destination? */
    }

    if ( destptr<srcptr || destptr>=(srcptr+n) )
    {
        /*
         * If blocks do not overlap or or backwards copy is requested,
         * it is safe to copy the source block from begin to end.
         */
        while (n--)
        {
            *destptr++ = *srcptr++;
        }
    }
    else
    {
        /*
         * If forward copy is requested and blocks overlap, forward copy
         * (from block's begin to end) would cause a corruption.
         * Hence backward copy (from end to begin) is performed.
         */
        srcptr += n - 1;
        destptr += n - 1;

        while (n--)
        {
            *destptr-- = *srcptr--;
        }
    }

    return destination;
}


/**
 * Copy string.
 *
 * Copies the C string pointed by 'source' into the array pointed by
 * 'destination', including the terminating null character (and stopping
 * at that point).
 *
 * To avoid overflows, the size of the array pointed by destination shall be
 * long enough to contain the same C string as source (including the
 * terminating null character), and should not overlap in memory with source.
 *
 * @param destination - pointer to the destination array where the content is to be copied
 * @param source - C string to be copied
 *
 * @return 'destination' is returned or NULL if any parameter equals NULL
 */
char* strcpy(char* destination, const char* source)
{
    const char* srcptr = source;
    char* destptr = destination;

    /* sanity check */
    if ( NULL==destination || NULL==source )
    {
        return NULL;
    }

    while ( '\0' != *srcptr )
    {
        *destptr++ = *srcptr++;
    }

    /* Do not forget to append a '\0' at the end of destination! */
    *destptr = '\0';

    return destination;
}


/**
 * Get string length.
 *
 * Returns the length of the C string str.
 *
 * The length of a C string is determined by the terminating
 * null-character: A C string is as long as the number of
 * characters between the beginning of the string and the
 * terminating null character (without including the terminating
 * null character itself).
 *
 * If 'str' equals NULL, -1 (casted to sze_t) is returned.
 *
 * @param str - C string.
 *
 * @return the length of string
 */
size_t strlen(const char* str)
{
    const char* pc;

    /* sanity check: */
    if ( NULL == str )
    {
        return (size_t) -1;
    }

    /* Find the first occurrence of '\0' */
    for ( pc = str; '\0'!=*pc; ++pc );

    return (size_t) (pc - str);
}


/**
 * Concatenate strings.
 *
 * Appends a copy of the source string to the destination string.
 * The terminating null character in 'destination' is overwritten
 * by the first character of 'source', and a null-character is included
 * at the end of the new string formed by the concatenation of both
 * in 'destination'.
 *
 * 'destination' and 'source' shall not overlap.
 *
 * Nothing is done if either 'destination' or 'source' equals NULL.
 *
 * @param destination - pointer to the destination array, which should contain a
 *        C string, and be large enough to contain the concatenated resulting string
 * @param source - C string to be appended, it should not overlap 'destination'
 *
 * @return 'destination' is returned
 */
char* strcat(char* destination, const char* source)
{
    char* pd;
    const char* ps;

    /* sanity check */
    if ( NULL == destination || NULL==source )
    {
        return destination;
    }

    /* Find the first occurrence of '\0' in 'destination */
    for ( pd=destination; '\0'!=*pd; ++pd );

    /*
     * Append characters of 'source' to it until
     * the first occurrence of '\0' in 'source'
     */
    for ( ps=source; '\0'!=*ps; ++ps )
    {
        *(pd++) = *ps;
    }

    /* And append '\0' to 'destination': */
    *pd = '\0';

    return destination;
}


/**
 * Compare two strings.
 *
 * Compares the C string 'str1' to the C string 'str2'.
 *
 * This function starts comparing the first character of each
 * string. If they are equal to each other, it continues with
 * the following pairs until the characters differ or until a
 * terminating null-character is reached.
 *
 * This function performs a binary comparison of the characters
 * and does not take into account any locale-specific rules.
 *
 * 0 is returned if any string equals NULL.
 *
 * @param str1 - C string to be compared
 * @param str2 - C string to be compared
 *
 * @return an integral value indicating the relationship between
 *         the strings: a zero value indicates that both strings
 *         are equal, a value greater than zero indicates that the
 *         first character that does not match has a greater value
 *         in 'str1' than in 'str2'; and a value less than zero
 *         indicates the opposite
 */
int32_t strcmp(const char* str1, const char* str2)
{
    const char* s1 = str1;
    const char* s2 = str2;

    /* sanity check: */
    if ( NULL==str1 || NULL==str2 )
    {
        return 0;
    }

    /* Find the first occurrence of '\0' or different characters: */
    while ( ( '\0' != *s1 && '\0' != *s2 ) && (*s1 == *s2) )
    {
        ++s1;
        ++s2;
    }

    /*
     * To prevent issues with difference of signed values,
     * cast both characters to unsigned char:
     */
    return (int32_t) ( *((const unsigned char*) s1) - *((const unsigned char*) s2) );
}


/**
 * Convert integer to string.
 *
 * Converts an integer value to a null-terminated string using
 * the specified base 'radix' and stores the result in the array
 *  given by 'str' parameter.
 *
 * If 'radix' is 10 and value is negative, the resulting string is
 * preceded with a minus sign (-). With any other 'radix', value is
 * always considered unsigned.
 *
 * 'str' should be an array long enough to contain any possible value:
 * (sizeof(int)*8+1) for radix=2, i.e. 17 bytes in 16-bits platforms
 * and 33 in 32-bits platforms.
 *
 * Nothing will be done if 'str' is NULL or 'radix' is out of the
 * valid range (between 2 and 36).
 *
 * @note This function is not defined in ANSI-C and not all toolchains
 *       include its declaration in standard headers. If this is the case
 *       with your selected toolchain, just copy the function's
 *       declaration below.
 *
 * @param value - value to be converted to a string
 * @param str - array in memory where to store the resulting null-terminated string
 * @param radix - numerical base used to represent the value as a string, between 2 and 36, where 10 means decimal base, 16 hexadecimal, 8 octal, and 2 binary
 *
 * @return A pointer to the resulting null-terminated string, same as parameter 'str'
 */
char* itoa(int32_t value, char* str, uint8_t radix)
{
    uint8_t mod;
    uint8_t neg = 0;
    char* pos = str;
    char* s = str;
    char q;
    uint32_t val;

    /* sanity check: */
    if ( NULL==str || radix<2 || radix>36 )
    {
        return str;
    }

    /* In decimal base, negative values are allowed */
    if ( 10==radix && value<0 )
    {
        val = -value;
        neg = 1;
    }
    else
    {
        val = value;
    }

    /*
     * Characters will be entered into 'str' in reverse order.
     * Later the order will be inversed.
     */
    do
    {
        /* Current digit is simply calculated as remainder... */
        mod = val % radix;
        /* ...and displayed appropriately */
        *(pos++) = ( mod<10 ? '0' + mod : 'a' + mod - 10 );
        /* Finally 'va;' must be updated */
        val /= radix;
    }
    while ( 0 != val );

    /* Append the negative sign if necessary */
    if (neg)
    {
        *(pos++) = '-';
    }

    /*
     * And append the string terminator.
     * Note that a pointer to the last character
     * (without the terminator) will be used to
     * invert the string.
     */
    *(pos--) = '\0';

    /*
     * Finally invert the string by using two pointers,
     * one incrementing from the start and the other one
     * decrementing from the end of the string
     * ('\0' terminator not included!).
     */
    for ( s=str; s<pos; ++s, --pos)
    {
        q = *s;
        *s = *pos;
        *pos = q;
    }

    return str;
}
