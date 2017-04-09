/*
 * Created by Jeremy Weed
 * jweed262@umd.edu
 */

#ifndef RING_BUFFER_H
#define RING_BUFFER_H

#include <stdint.h>
#include <FreeRTOS.h>
#include <sys/types.h>


#define rbSUCCSES 0
#define rbFAIL 1
#define rbNULL 2

typedef struct _RingBuffer RingBuffer;

RingBuffer *createRingBuffer(uint16_t length, uint8_t size);

RingBuffer freeRingBuffer(RingBuffer** ringBuffer);

int pop(RingBuffer* ringBuffer, void* data);

int push(RingBuffer* ringBuffer, void* data);

int peek(RingBuffer* ringBuffer, void* data);

static int copy(void* from, void* to, ssize_t length);

#endif
