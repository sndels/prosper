#ifndef PROSPER_SCENE_MESH_HPP
#define PROSPER_SCENE_MESH_HPP

#include <cstdint>
#include <vulkan/vulkan.hpp>

constexpr vk::Format sVertexPositionFormat = vk::Format::eR16G16B16A16Sfloat;
constexpr uint32_t sVertexPositionByteSize = 8;
constexpr vk::Format sVertexNormalFormat = vk::Format::eA2B10G10R10SnormPack32;
constexpr vk::Format sVertexTangentFormat = vk::Format::eA2B10G10R10SnormPack32;
constexpr vk::Format sVertexTexCoord0Format = vk::Format::eR16G16Sfloat;

struct GeometryMetadata
{
    uint32_t bufferIndex{0xFFFF'FFFF};
    // These offsets are into the geometry data buffers. Most are for U32/F32
    // and an offset of 0xFFFF'FFFF signals an unused attribute.
    // This addresses U16 if short indices are in use.
    uint32_t indicesOffset{0xFFFF'FFFF};
    uint32_t positionsOffset{0xFFFF'FFFF};
    uint32_t normalsOffset{0xFFFF'FFFF};
    uint32_t tangentsOffset{0xFFFF'FFFF};
    uint32_t texCoord0sOffset{0xFFFF'FFFF};
    uint32_t meshletsOffset{0xFFFF'FFFF};
    uint32_t meshletBoundsOffset{0xFFFF'FFFF};
    // This addresses U16 if short indices are in use.
    uint32_t meshletVerticesOffset{0xFFFF'FFFF};
    // This addresses U8.
    uint32_t meshletTrianglesByteOffset{0xFFFF'FFFF};
    uint32_t usesShortIndices{0};
};
// These are uploaded onto the gpu and tight packing is assumed
static_assert(alignof(GeometryMetadata) == sizeof(uint32_t));

struct MeshInfo
{
    uint32_t vertexCount{0};
    uint32_t indexCount{0};
    uint32_t meshletCount{0};
    uint32_t materialID{0};
};

#endif // PROSPER_SCENE_MESH_HPP
