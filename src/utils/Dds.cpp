#include "Dds.hpp"

#include "Utils.hpp"

#include <fstream>
#include <iostream>

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
    Texture3d = 4,
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

const uint32_t sDdsMagic = 0x2053'4444;
const uint32_t sDx10Magic = 0x3031'5844;

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
    wheels::Allocator &alloc, uint32_t width, uint32_t height, uint32_t depth,
    DxgiFormat format, uint32_t mipLevelCount)
// NOLINTEND(bugprone-easily-swappable-parameters)
: width{width}
, height{height}
, depth{depth}
, format{format}
, mipLevelCount{mipLevelCount}
, data{alloc}
, levelByteOffsets{alloc}
{
    const uint32_t levelCount = std::max(mipLevelCount, depth);
    uint32_t totalByteSize = 0;
    if (depth <= 1)
    {
        levelByteOffsets.reserve(levelCount);

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
                WHEELS_ASSERT(
                    levelWidth % 4 == 0 && levelHeight % 4 == 0 &&
                    "BC7 mips should be divide evenly by 4x4");
                WHEELS_ASSERT(
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
    }
    else
    {
        WHEELS_ASSERT(
            mipLevelCount == 1 &&
            "Volume textures with mips are not implemented");
        switch (format)
        {
        case DxgiFormat::R9G9B9E5SharedExp:
            totalByteSize = width * height * depth * 4;
            break;
        default:
            throw std::runtime_error("Unimplemented DxgiFormat");
        }
    }
    data.resize(totalByteSize);
}

void writeDds(const Dds &dds, const std::filesystem::path &path)
{
    std::filesystem::remove(path);

    // Write into a tmp file and rename when done to minimize the potential for
    // corrupted files
    std::filesystem::path tmpPath = path;
    tmpPath.replace_extension("dds_TMP");
    // NOTE:
    // Caches aren't supposed to be portable so this doesn't pay attention to
    // endianness.
    std::ofstream outFile{tmpPath, std::ios_base::binary};

    writeRaw(outFile, sDdsMagic);

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
        0x000A'1007u
        // mipmapcount | uncompressed | required
        : 0x0002'100Fu;
    // clang-format on

    WHEELS_ASSERT(
        dds.depth == 1 && "DDS writes for 3D textures are not implemented");

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
                .dwRBitMask = pixelBits == 32 ? 0x0000'00FF : 0u,
                .dwGBitMask = pixelBits == 32 ? 0x0000'FF00 : 0u,
                .dwBBitMask = pixelBits == 32 ? 0x00FF'0000 : 0u,
                .dwABitMask = pixelBits == 32 ? 0xFF00'0000 : 0u,
            },
        // gli had mipmaps tagged even for textures that had 1 mipmap, let's
        // match
        // MIPMAP | TEXTURE
        .dwCaps = 0x00401000u,
    };
    writeRaw(outFile, ddsHeader);

    const DdsHeaderDxt10 ddsHeaderDxt10{
        .dxgiFormat = dds.format,
        .resourceDimension = D3d10ResourceDimension::Texture2d,
        .arraySize = 1,
    };
    writeRaw(outFile, ddsHeaderDxt10);

    writeRawSpan(outFile, dds.data.span());

    outFile.close();

    // Make sure we have rw permissions for the user to be nice
    const std::filesystem::perms initialPerms =
        std::filesystem::status(tmpPath).permissions();
    std::filesystem::permissions(
        tmpPath, initialPerms | std::filesystem::perms::owner_read |
                     std::filesystem::perms::owner_write);

    // Rename when the file is done to minimize the potential of a corrupted
    // file
    std::filesystem::rename(tmpPath, path);
}

Dds readDds(Allocator &alloc, const std::filesystem::path &path)
{
    // NOTE:
    // Caches aren't supposed to be portable so this doesn't pay attention to
    // endianness.
    std::ifstream inFile{path, std::ios_base::binary};

    uint32_t magic = 0;
    readRaw(inFile, magic);
    WHEELS_ASSERT(magic == sDdsMagic && "File doesn't appear to be a dds");

    DdsHeader ddsHeader;
    readRaw(inFile, ddsHeader);

    WHEELS_ASSERT(ddsHeader.dwSize == 124 && "Unexpexted DDS_HEADER size");
    // Programming guide advises against checking 0x1, 0x1000 and 0x2000, but
    // gli was pedantic here so let's do that as well. This is for our cache
    // after all...
    // https://learn.microsoft.com/en-us/windows/win32/direct3ddds/dds-header
    WHEELS_ASSERT(
        ((ddsHeader.dwFlags == 0x0002'100F) ||
         (ddsHeader.dwFlags == 0x0080'100F) ||
         (ddsHeader.dwFlags == 0x000A'1007)) &&
        "Unexpexted DDS_FLAGS ");
    WHEELS_ASSERT(
        ddsHeader.ddspf.dwSize == 32 && "Unexpexted DDS_PIXEL_FORMAT size");
    WHEELS_ASSERT(ddsHeader.ddspf.dwFlags == 0x4 && "Expected valid FourCC");
    WHEELS_ASSERT(
        ddsHeader.ddspf.dwFourCC == sDx10Magic && "Expected a Dx10 header");
    WHEELS_ASSERT(
        (ddsHeader.ddspf.dwRGBBitCount == 32 ||
         ddsHeader.ddspf.dwRGBBitCount == 0) &&
        "Expected a 32bit format or 0");
    WHEELS_ASSERT(
        (ddsHeader.ddspf.dwRBitMask == 0x0000'00FF ||
         ddsHeader.ddspf.dwRBitMask == 0) &&
        "Expected R bit mask 0x0000'00FF or 0");
    WHEELS_ASSERT(
        (ddsHeader.ddspf.dwGBitMask == 0x0000'FF00 ||
         ddsHeader.ddspf.dwGBitMask == 0) &&
        "Expected G bit mask 0x0000'FF00 or 0");
    WHEELS_ASSERT(
        (ddsHeader.ddspf.dwBBitMask == 0x00FF'0000 ||
         ddsHeader.ddspf.dwBBitMask == 0) &&
        "Expected B bit mask 0x00FF'0000 or 0");
    WHEELS_ASSERT(
        (ddsHeader.ddspf.dwABitMask == 0xFF00'0000 ||
         ddsHeader.ddspf.dwABitMask == 0) &&
        "Expected A bit mask 0xFF00'0000 or 0");
    // gli had mipmaps tagged even for textures that had 1 mipmap, let's match
    // https://learn.microsoft.com/en-us/windows/win32/direct3ddds/dds-header
    WHEELS_ASSERT(
        (ddsHeader.dwCaps == 0x0040'1000 || ddsHeader.dwCaps == 0x0000'1008) &&
        "Unexpected DDS_CAPS");

    DdsHeaderDxt10 ddsHeaderDxt10;
    readRaw(inFile, ddsHeaderDxt10);

    WHEELS_ASSERT(
        (ddsHeaderDxt10.dxgiFormat == DxgiFormat::R8G8B8A8Unorm ||
         ddsHeaderDxt10.dxgiFormat == DxgiFormat::R9G9B9E5SharedExp ||
         ddsHeaderDxt10.dxgiFormat == DxgiFormat::BC7Unorm) &&
        "Only R8G8B8A8Unorm and BC7Unorm DDS textures are supported");
    WHEELS_ASSERT(
        (ddsHeaderDxt10.resourceDimension ==
             D3d10ResourceDimension::Texture2d ||
         ddsHeaderDxt10.resourceDimension ==
             D3d10ResourceDimension::Texture3d) &&
        "Only Texture2d and Texture3d DDS resource dimension is supported");
    WHEELS_ASSERT(
        ddsHeaderDxt10.arraySize == 1 &&
        "DDS texture arrays are not supported");

    Dds ret{
        alloc,
        ddsHeader.dwWidth,
        ddsHeader.dwHeight,
        ddsHeader.dwDepth,
        ddsHeaderDxt10.dxgiFormat,
        ddsHeader.dwMipMapCount};

    readRawSpan(inFile, ret.data.mut_span());

    inFile.read(
        reinterpret_cast<char *>(ret.data.data()),
        asserted_cast<std::streamsize>(ret.data.size()));

    return ret;
}
