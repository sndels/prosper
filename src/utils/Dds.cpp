#include "Dds.hpp"

#include <fstream>
#include <iostream>
#include <string>

using namespace wheels;

namespace
{

// https://learn.microsoft.com/en-us/windows/win32/direct3ddds/dds-pixelformat
struct DdsPixelFormat
{
    uint32_t dwSize{32};
    uint32_t dwFlags{0};
    uint32_t dwFourCC{0};
    uint32_t dwRGBBitCount{0};
    uint32_t dwRBitMask{0};
    uint32_t dwGBitMask{0};
    uint32_t dwBBitMask{0};
    uint32_t dwABitMask{0};
};

// https://learn.microsoft.com/en-us/windows/win32/direct3ddds/dds-header
struct DdsHeader
{
    uint32_t dwSize{124};
    uint32_t dwFlags{0};
    uint32_t dwHeight{0};
    uint32_t dwWidth{0};
    uint32_t dwPitchOrLinearSize{0};
    uint32_t dwDepth{0};
    uint32_t dwMipMapCount{0};
    // NOLINTNEXTLINE(*-avoid-c-arrays): Match the original docs
    uint32_t dwReserved1[11]{};
    DdsPixelFormat ddspf{0};
    uint32_t dwCaps{0};
    uint32_t dwCaps2{0};
    uint32_t dwCaps3{0};
    uint32_t dwCaps4{0};
    uint32_t dwReserved2{0};
};

// https://learn.microsoft.com/en-us/windows/win32/api/d3d10/ne-d3d10-d3d10_resource_dimension
enum class D3d10ResourceDimension
{
    Unknown = 0,
    Texture2d = 3,
};

// https://learn.microsoft.com/en-us/windows/win32/direct3ddds/dds-header-dxt10
struct DdsHeaderDxt10
{
    DxgiFormat dxgiFormat{DxgiFormat::Unknown};
    D3d10ResourceDimension resourceDimension{D3d10ResourceDimension::Unknown};
    uint32_t miscFlag{0};
    uint32_t arraySize{0};
    uint32_t miscFlags2{0};
};

const uint32_t sDdsMagic = 0x20534444;
const uint32_t sDx10Magic = 0x30315844;

bool isFormatCompressed(DxgiFormat format)
{
    switch (format)
    {
    case DxgiFormat::R8G8B8A8Unorm:
        return false;
    case DxgiFormat::BC7Unorm:
        return true;
    default:
        break;
    }
    throw std::runtime_error("Unkown DxgiFormat");
}

} // namespace

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
Dds::Dds(
    wheels::Allocator &alloc, uint32_t width, uint32_t height,
    DxgiFormat format, uint32_t mipLevelCount)
// NOLINTEND(bugprone-easily-swappable-parameters)
: width{width}
, height{height}
, format{format}
, mipLevelCount{mipLevelCount}
, data{alloc}
, levelByteOffsets{alloc}
{
    uint32_t totalByteSize = 0;
    levelByteOffsets.reserve(mipLevelCount);
    for (uint32_t i = 0; i < mipLevelCount; ++i)
    {
        uint32_t levelWidth = 0;
        uint32_t levelHeight = 0;
        uint32_t levelByteSize = 0;
        switch (format)
        {
        case DxgiFormat::R8G8B8A8Unorm:
            levelWidth = std::max(width >> i, 1u);
            levelHeight = std::max(height >> i, 1u);
            levelByteSize = levelWidth * levelHeight * 4;
            break;
        case DxgiFormat::BC7Unorm:
            // Each 4x4 block is 16bytes
            levelWidth = std::max(width >> i, 1u);
            levelHeight = std::max(height >> i, 1u);
            assert(
                levelWidth % 4 == 0 && levelHeight % 4 == 0 &&
                "BC7 mips should be divide evenly by 4x4");
            assert(
                levelWidth >= 4 && levelHeight >= 4 &&
                "BC7 mip dimensions should be at least 4x4");
            levelByteSize = levelWidth / 4 * levelHeight / 4 * 16;
            break;
        default:
            throw std::runtime_error("Unknown DxgiFormat");
        }

        levelByteOffsets.push_back(totalByteSize);
        totalByteSize += levelByteSize;
    }
    data.resize(totalByteSize);
}

void writeDds(const Dds &dds, const std::filesystem::path &path)
{
    // NOTE:
    // Caches aren't supposed to be portable so this doesn't pay attention to
    // endianness.
    std::ofstream outFile{path, std::ios_base::binary};
    outFile.write(
        reinterpret_cast<const char *>(&sDdsMagic), sizeof(sDdsMagic));

    const bool isCompressed = isFormatCompressed(dds.format);
    const uint32_t pixelStride = isCompressed ? 0 : 4;
    const uint32_t pixelBits = isCompressed ? 0 : 32;
    const uint32_t pitchOrLinearSize =
        isCompressed ? (dds.mipLevelCount == 1
                            ? dds.levelByteOffsets[0]
                            : dds.levelByteOffsets[1] - dds.levelByteOffsets[0])
                     : dds.width * dds.height * pixelStride;

    // clang-format off
    const uint32_t flags = isCompressed ?
        // compressed | mipmapcount | required
        0x000A1007u
        // mipmapcount | uncompressed | required
        : 0x0002100Fu;
    // clang-format on
    // We don't use the legacy header so write it empty
    const DdsHeader ddsHeader{
        .dwFlags = flags,
        .dwHeight = dds.height,
        .dwWidth = dds.width,
        .dwPitchOrLinearSize = pitchOrLinearSize,
        .dwMipMapCount = dds.mipLevelCount,
        .ddspf =
            DdsPixelFormat{
                .dwFlags = 0x4, // FourCC
                .dwFourCC = sDx10Magic,
                .dwRGBBitCount = pixelBits,
                .dwRBitMask = pixelBits == 32 ? 0x000000FF : 0u,
                .dwGBitMask = pixelBits == 32 ? 0x0000FF00 : 0u,
                .dwBBitMask = pixelBits == 32 ? 0x00FF0000 : 0u,
                .dwABitMask = pixelBits == 32 ? 0xFF000000 : 0u,
            },
        // gli had mipmaps tagged even for textures that had 1 mipmap, let's
        // match
        // MIPMAP | TEXTURE
        .dwCaps = 0x00401000u,
    };
    outFile.write(
        reinterpret_cast<const char *>(&ddsHeader), sizeof(ddsHeader));

    const DdsHeaderDxt10 ddsHeaderDxt10{
        .dxgiFormat = dds.format,
        .resourceDimension = D3d10ResourceDimension::Texture2d,
        .arraySize = 1,
    };
    outFile.write(
        reinterpret_cast<const char *>(&ddsHeaderDxt10),
        sizeof(ddsHeaderDxt10));

    outFile.write(
        reinterpret_cast<const char *>(dds.data.data()), dds.data.size());
    outFile.close();

    // Make sure we have rw permissions for the user to be nice
    const std::filesystem::perms initialPerms =
        std::filesystem::status(path).permissions();
    std::filesystem::permissions(
        path, initialPerms | std::filesystem::perms::owner_read |
                  std::filesystem::perms::owner_write);
}

Dds readDds(Allocator &alloc, const std::filesystem::path &path)
{
    // NOTE:
    // Caches aren't supposed to be portable so this doesn't pay attention to
    // endianness.
    std::ifstream inFile{path, std::ios_base::binary};
    uint32_t magic = 0;
    inFile.read(reinterpret_cast<char *>(&magic), sizeof(magic));
    assert(magic == sDdsMagic && "File doesn't appear to be a dds");

    // We don't use the legacy header so write it empty
    DdsHeader ddsHeader;
    inFile.read(reinterpret_cast<char *>(&ddsHeader), sizeof(ddsHeader));
    assert(ddsHeader.dwSize == 124 && "Unexpexted DDS_HEADER size");
    // Programming guide advises against checking 0x1, 0x1000 and 0x2000, but
    // gli was pedantic here so let's do that as well. This is for our cache
    // after all...
    // mipmapcount | uncompressed | required
    assert(
        (ddsHeader.dwFlags == 0x0002100F) ||
        (ddsHeader.dwFlags == 0x000A1007) && "Unexpexted DDS_FLAGS ");
    assert(ddsHeader.ddspf.dwSize == 32 && "Unexpexted DDS_PIXEL_FORMAT size");
    assert(ddsHeader.ddspf.dwFlags == 0x4 && "Expected valid FourCC");
    assert(ddsHeader.ddspf.dwFourCC == sDx10Magic && "Expected a Dx10 header");
    assert(
        (ddsHeader.ddspf.dwRGBBitCount == 32 ||
         ddsHeader.ddspf.dwRGBBitCount == 0) &&
        "Expected a 32bit format or 0");
    assert(
        (ddsHeader.ddspf.dwRBitMask == 0x000000FF ||
         ddsHeader.ddspf.dwRBitMask == 0) &&
        "Expected R bit mask 0x000000FF or 0");
    assert(
        (ddsHeader.ddspf.dwGBitMask == 0x0000FF00 ||
         ddsHeader.ddspf.dwGBitMask == 0) &&
        "Expected G bit mask 0x0000FF00 or 0");
    assert(
        (ddsHeader.ddspf.dwBBitMask == 0x00FF0000 ||
         ddsHeader.ddspf.dwBBitMask == 0) &&
        "Expected B bit mask 0x00FF0000 or 0");
    assert(
        (ddsHeader.ddspf.dwABitMask == 0xFF000000 ||
         ddsHeader.ddspf.dwABitMask == 0) &&
        "Expected A bit mask 0xFF000000 or 0");
    // gli had mipmaps tagged even for textures that had 1 mipmap, let's match
    assert(ddsHeader.dwCaps == 0x00401000 && "Unexpected DDS_CAPS");

    DdsHeaderDxt10 ddsHeaderDxt10;
    inFile.read(
        reinterpret_cast<char *>(&ddsHeaderDxt10), sizeof(ddsHeaderDxt10));
    assert(
        (ddsHeaderDxt10.dxgiFormat == DxgiFormat::R8G8B8A8Unorm ||
         ddsHeaderDxt10.dxgiFormat == DxgiFormat::BC7Unorm) &&
        "Only R8G8B8A8Unorm and BC7Unorm DDS textures are supported");
    assert(
        ddsHeaderDxt10.resourceDimension == D3d10ResourceDimension::Texture2d &&
        "Only Texture2d DDS resource dimension is supported");
    assert(
        ddsHeaderDxt10.arraySize == 1 &&
        "DDS texture arrays are not supported");

    Dds ret{
        alloc, ddsHeader.dwWidth, ddsHeader.dwHeight, ddsHeaderDxt10.dxgiFormat,
        ddsHeader.dwMipMapCount};

    inFile.read(reinterpret_cast<char *>(ret.data.data()), ret.data.size());

    return ret;
}
