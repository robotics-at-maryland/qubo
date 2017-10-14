#include <pneumatics.h>

const Transaction tPneumaticsSet = {
    .name = "Pneumatics Set",
    .id = M_ID_PNEUMATICS_SET,
    .request = sizeof(struct Pneumatics_Set),
    .response = EMPTY,
};

const Error ePneumaticsUnreachable = {
    .name = "Pneumatics Unreachable",
    .id = E_ID_PNEUMATICS_UNREACHABLE,
    .size = EMPTY,
};
