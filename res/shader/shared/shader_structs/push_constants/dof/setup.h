#ifndef SHADER_STRUCTS_PUSH_CONSTANTS_DOF_SETUP_H
#define SHADER_STRUCTS_PUSH_CONSTANTS_DOF_SETUP_H

#include "../../fields.h"

struct SetupPC
{
    STRUCT_FIELD(float, focusDistance, 0.f);
    STRUCT_FIELD(float, maxBackgroundCoC, 0.f);
    STRUCT_FIELD(float, maxCoC, 0.f);
};

#endif // SHADER_STRUCTS_PUSH_CONSTANTS_DOF_SETUP_H
