#include <safety.h>

const Transaction tSafetyStatus = {
    .name = "Safety Status",
    .id = M_ID_SAFETY_STATUS,
    .request = EMPTY,
    .response = sizeof(struct Safety_Status),
};

const Transaction tSafetySetSafe = {
    .name = "Safety Set Safe",
    .id = M_ID_SAFETY_SET_SAFE,
    .request = EMPTY,
    .response = EMPTY,
};

const Transaction tSafetySetUnsafe = {
    .name = "Safety Set Unsafe",
    .id = M_ID_SAFETY_SET_UNSAFE,
    .request = EMPTY,
    .response = EMPTY,
};
