#ifndef PROSPER_UTILS_KTX_HPP
#define PROSPER_UTILS_KTX_HPP

#include <cstdint>
#include <filesystem>
#include <vulkan/vulkan.hpp>
#include <wheels/allocators/allocator.hpp>
#include <wheels/containers/array.hpp>

// https://registry.khronos.org/KTX/specs/2.0/ktxspec.v2.html
struct Ktx
{
    uint32_t width{0};
    uint32_t height{0};
    uint32_t depth{0};
    vk::Format format{vk::Format::eUndefined};
    uint32_t arrayLayerCount{0};
    uint32_t faceCount{0};
    uint32_t mipLevelCount{0};
    wheels::Array<uint8_t> data;
    // Offsets for individual faces in the texture. Indexed using
    // (iMip * arrayLayerCount * faceCount) + (iLayer * faceCount) + iFace
    wheels::Array<uint32_t> levelByteOffsets;
};

Ktx readKtx(wheels::Allocator &alloc, const std::filesystem::path &path);

#endif // PROSPER_UTILS_KTX_HPP
