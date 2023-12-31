#ifndef PROSPER_SCENE_MESH_HPP
#define PROSPER_SCENE_MESH_HPP

#include "../gfx/Resources.hpp"

struct MeshBuffers
{
    uint32_t bufferIndex{0xFFFFFFFF};
    // All of these offsets are into the data interpreted as a u32 'array'
    // offset of 0xFFFFFFFF signals
    uint32_t indicesOffset{0xFFFFFFFF};
    uint32_t positionsOffset{0xFFFFFFFF};
    uint32_t normalsOffset{0xFFFFFFFF};
    uint32_t tangentsOffset{0xFFFFFFFF};
    uint32_t texCoord0sOffset{0xFFFFFFFF};
    uint32_t usesShortIndices{0};
};
// These are uploaded onto the gpu and tight packing is assumed
static_assert(alignof(MeshBuffers) == sizeof(uint32_t));

struct MeshInfo
{
    uint32_t vertexCount{0};
    uint32_t indexCount{0};
    uint32_t materialID{0};
};

#endif // PROSPER_SCENE_MESH_HPP
