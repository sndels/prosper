#ifndef PROSPER_MESH_HPP
#define PROSPER_MESH_HPP

#include "Device.hpp"
#include "Material.hpp"

struct MeshBuffers
{
    struct Buffer
    {
        uint32_t index{0xFFFFFFFF};
        uint32_t offset{0};
    };

    Buffer indices;
    Buffer positions;
    Buffer normals;
    Buffer tangents;
    Buffer texCoord0s;
    uint32_t usesShortIndices{0};
};
// These are uploaded onto the gpu and tight packing is assumed
static_assert(sizeof(MeshBuffers) == (5 * 2 + 1) * sizeof(uint32_t));
static_assert(alignof(MeshBuffers) == sizeof(uint32_t));

struct MeshInfo
{
    uint32_t vertexCount{0};
    uint32_t indexCount{0};
    uint32_t materialID{0};
};

#endif // PROSPER_MESH_HPP
