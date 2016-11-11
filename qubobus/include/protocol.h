#include <stdint.h>
#include <modules.h>

#ifndef QUBOBUS_PROTOCOL_H
#define QUBOBUS_PROTOCOL_H

enum {
    E_ID_PROTOCOL = M_ID_OFFSET_CORE,

    E_ID_CHECKSUM,

    E_ID_SEQUENCE,

    E_ID_TIMEOUT,
};

extern const Error eProtocol;
extern const Error eChecksum;
extern const Error eSequence;
extern const Error eTimeout;

#endif
