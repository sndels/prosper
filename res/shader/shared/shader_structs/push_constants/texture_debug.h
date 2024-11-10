#ifndef SHADER_STRUCTS_PUSH_CONSTANTS_TEXTURE_DEBUG_H
#define SHADER_STRUCTS_PUSH_CONSTANTS_TEXTURE_DEBUG_H

#include "../fields.h"

#define TEXTURE_DEBUG_RANGE_DEFAULT                                            \
    glm::vec2 { 0.f, 1.f }

struct TextureDebugPC
{
    STRUCT_FIELD_GLM(ivec2, inRes, {});
    STRUCT_FIELD_GLM(ivec2, outRes, {});

    STRUCT_FIELD_GLM(vec2, range, TEXTURE_DEBUG_RANGE_DEFAULT);
    STRUCT_FIELD_GLM(uint, lod, 0);
    STRUCT_FIELD_GLM(uint, flags, 0);

    STRUCT_FIELD_GLM(vec2, cursorUv, {});
};

#endif // SHADER_STRUCTS_PUSH_CONSTANTS_TEXTURE_DEBUG_H
