/*
 * Created by Jeremy Weed
 * jweed262@umd.edu
 */

#include "lib/include/ring_buffer.h"

struct _RingBuffer {
	void* buffer;
	uint16_t length;
	uint8_t size;
	uint16_t front;
	uint16_t back;
};

RingBuffer *createRingBuffer(uint16_t length, uint8_t size){
	RingBuffer* r_buf = (RingBuffer*) pvPortMalloc(sizeof(RingBuffer));
	r_buf->buffer = pvPortMalloc((length + 1) * (size));
	r_buf->length = length + 1;
	r_buf->size = size;
	r_buf->front = 0;
	r_buf->back = 0 ;
	return r_buf;
}

RingBuffer freeRingBuffer(RingBuffer** ringBuffer){
	if(*ringBuffer){
		if((*ringBuffer)->buffer){
			vPortFree((*ringBuffer)->buffer);
			(*ringBuffer)->buffer = NULL;
		}
		vPortFree(*ringBuffer);
		*ringBuffer = NULL;
	}
}

int pop(RingBuffer* ringBuffer, void* data){
	if(ringBuffer == NULL){
		return rbNULL;
	}
	if(ringBuffer->front == ringBuffer->back){
		//this means there isn't anything left in the buffer
		return rbFAIL;
	}
	//get the value, then push the front pointer back
	copy(&(ringBuffer->buffer[ringBuffer->front * ringBuffer->size]), data, ringBuffer->size);
	ringBuffer->front = (ringBuffer->front + 1) % ringBuffer->length;
	return rbSUCCSES;
}

int push(RingBuffer* ringBuffer, void* data){
	if(ringBuffer == NULL){
		return rbNULL;
	}
	copy(data, &(ringBuffer->buffer[ringBuffer->back * ringBuffer->size]), ringBuffer->size);
	//push the end backwards
	ringBuffer->back = (ringBuffer->back + 1) % ringBuffer->length;
	//push back the front if we just overwrote it
	if(ringBuffer->front == ringBuffer->back){
		ringBuffer->front = (ringBuffer->front + 1) % ringBuffer->length;
	}
	return rbSUCCSES;
}

int peek(RingBuffer* ringBuffer, void* data){
	if(ringBuffer == NULL){
		return rbNULL;
	}
	if(ringBuffer->front == ringBuffer->back){
		//this means there isn't anything left in the buffer
		return rbFAIL;
	}
	//get the value, then push the front pointer back
	copy(&(ringBuffer->buffer[ringBuffer->front * ringBuffer->size]), data, ringBuffer->size);
	return rbSUCCSES;
}

static int copy(void* from, void* to, ssize_t length){
	if(NULL == from || NULL == to){
		return rbNULL;
	}
	uint8_t* to_b = to;
	uint8_t* from_b = from;
	for(int i = 0; i < length; i++){
		to_b[i] = from_b[i];
	}
	return rbSUCCSES;
}
