#include <embedded.h>

const Transaction tEmbeddedStatus = {
    .name = "Embedded Status",
    .id = M_ID_EMBEDDED_STATUS,
    .request = EMPTY,
    .response = sizeof(struct Embedded_Status),
};

const Error eEmbeddedError = {
    .name = "Embedded Error",
    .id = E_ID_EMBEDDED_ERROR,
    .size = EMPTY,
};
