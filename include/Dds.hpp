#ifndef PROSPER_DDS_HPP
#define PROSPER_DDS_HPP

#include <cstdint>
#include <filesystem>

#include <wheels/allocators/allocator.hpp>
#include <wheels/containers/array.hpp>

// https://learn.microsoft.com/en-us/windows/win32/api/dxgiformat/ne-dxgiformat-dxgi_format
enum class DxgiFormat
{
    Unknown = 0,
    R8G8B8A8Unorm = 28,
};

// This is a pretty simple format so let's just do the support ourselves
// https://learn.microsoft.com/en-us/windows/win32/direct3ddds/dx-graphics-dds-pguide
struct Dds
{
    uint32_t width{0};
    uint32_t height{0};
    DxgiFormat format{DxgiFormat::Unknown};
    uint32_t mipLevelCount{0};
    wheels::Array<uint8_t> data;
    wheels::Array<uint32_t> levelByteOffsets;
};

void writeDds(const Dds &dds, const std::filesystem::path &path);
Dds readDds(wheels::Allocator &alloc, const std::filesystem::path &path);

#endif // PROSPER_DDS_HPP
