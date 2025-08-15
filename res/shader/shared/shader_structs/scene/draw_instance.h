#ifndef SHARED_SHADER_STRUCTS_SCENE_DRAW_INSTANCE_H
#define SHARED_SHADER_STRUCTS_SCENE_DRAW_INSTANCE_H

#include "../fields.h"

#ifdef __cplusplus
namespace scene::shader_structs
{
#endif // __cplusplus

struct DrawInstance
{
    STRUCT_FIELD_GLM(uint, modelInstanceIndex, 0);
    STRUCT_FIELD_GLM(uint, meshIndex, 0xFFFF'FFFF);
    STRUCT_FIELD_GLM(uint, materialIndex, 0xFFFF'FFFF);
};

#ifdef __cplusplus

// These are uploaded onto the gpu and tight packing is assumed
static_assert(alignof(DrawInstance) == sizeof(uint32_t));

} // namespace scene::shader_structs

#endif // __cplusplus

#endif // SHARED_SHADER_STRUCTS_SCENE_DRAW_INSTANCE_H
