#include <protocol.h>

const Error eProtocol = {
    .name = "Protocol",
    .id = E_ID_PROTOCOL,
    .size = EMPTY,
};

const Error eChecksum = {
    .name = "Checksum",
    .id = E_ID_CHECKSUM,
    .size = EMPTY,
};

const Error eSequence = {
    .name = "Sequence",
    .id = E_ID_SEQUENCE,
    .size = EMPTY,
};

const Error eTimeout = {
    .name = "Timeout",
    .id = E_ID_TIMEOUT,
    .size = EMPTY,
};
