#ifndef SHARED_SHADER_STRUCTS_SCENE_MODEL_INSTANCE_TRANSFORMS_H
#define SHARED_SHADER_STRUCTS_SCENE_MODEL_INSTANCE_TRANSFORMS_H

#include "../fields.h"

#ifdef __cplusplus
namespace scene::shader_structs
{
#endif // __cplusplus

struct ModelInstanceTransforms
{
    STRUCT_FIELD_GLM(mat3x4, modelToWorld, 1.f);
    STRUCT_FIELD_GLM(mat3x4, normalToWorld, 1.f);
};

#ifdef __cplusplus
} //  namespace scene::shader_structs
#endif // __cplusplus

#endif // SHARED_SHADER_STRUCTS_SCENE_MODEL_INSTANCE_TRANSFORMS_H
