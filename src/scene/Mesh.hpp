#ifndef PROSPER_SCENE_MESH_HPP
#define PROSPER_SCENE_MESH_HPP

#include <cstdint>
#include <vulkan/vulkan.hpp>

constexpr vk::Format sVertexPositionFormat = vk::Format::eR16G16B16A16Sfloat;
constexpr uint32_t sVertexPositionByteSize = 8;
constexpr vk::Format sVertexNormalFormat = vk::Format::eA2B10G10R10SnormPack32;
constexpr vk::Format sVertexTangentFormat = vk::Format::eA2B10G10R10SnormPack32;
constexpr vk::Format sVertexTexCoord0Format = vk::Format::eR16G16Sfloat;

#include <shader_structs/scene/geometry_metadata.h>

struct MeshInfo
{
    uint32_t vertexCount{0};
    uint32_t indexCount{0};
    uint32_t meshletCount{0};
    uint32_t materialIndex{0};
};

#endif // PROSPER_SCENE_MESH_HPP
