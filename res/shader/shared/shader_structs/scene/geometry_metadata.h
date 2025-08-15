#ifndef SHARED_SHADER_STRUCTS_SCENE_GEOMETRY_METADATA_H
#define SHARED_SHADER_STRUCTS_SCENE_GEOMETRY_METADATA_H

#include "../fields.h"

#ifdef __cplusplus
namespace scene::shader_structs
{
#endif // __cplusplus

struct GeometryMetadata
{
    STRUCT_FIELD_GLM(uint, bufferIndex, 0xFFFF'FFFF);
    // These offsets are into the geometry data buffers. Most are for U32/F32
    // and an offset of 0xFFFF'FFFF signals an unused attribute.
    // This addresses U16 if short indices are in use.
    STRUCT_FIELD_GLM(uint, indicesOffset, 0xFFFF'FFFF);
    STRUCT_FIELD_GLM(uint, positionsOffset, 0xFFFF'FFFF);
    STRUCT_FIELD_GLM(uint, normalsOffset, 0xFFFF'FFFF);
    STRUCT_FIELD_GLM(uint, tangentsOffset, 0xFFFF'FFFF);
    STRUCT_FIELD_GLM(uint, texCoord0sOffset, 0xFFFF'FFFF);
    STRUCT_FIELD_GLM(uint, meshletsOffset, 0xFFFF'FFFF);
    STRUCT_FIELD_GLM(uint, meshletBoundsOffset, 0xFFFF'FFFF);
    // This addresses U16 if short indices are in use.
    STRUCT_FIELD_GLM(uint, meshletVerticesOffset, 0xFFFF'FFFF);
    // This addresses U8.
    STRUCT_FIELD_GLM(uint, meshletTrianglesByteOffset, 0xFFFF'FFFF);
    STRUCT_FIELD_GLM(uint, usesShortIndices, 0);
};

#ifdef __cplusplus

// These are uploaded onto the gpu and tight packing is assumed
static_assert(alignof(GeometryMetadata) == sizeof(uint32_t));

} //  namespace scene::shader_structs

#endif // __cplusplus

#endif // SHARED_SHADER_STRUCTS_SCENE_GEOMETRY_METADATA_H
