#include "Ktx.hpp"

#include "Utils.hpp"
#include <cstdint>
#include <fstream>
#include <wheels/assert.hpp>
#include <wheels/containers/static_array.hpp>

using namespace wheels;

// Based on the official specs
// https://registry.khronos.org/KTX/specs/1.0/ktxspec.v1.html
// https://registry.khronos.org/KTX/specs/2.0/ktxspec.v2.html

namespace
{

// «KTX 20»\r\n\x1A\n
const StaticArray<uint8_t, 12> sFileIdentifier20{
    {0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A}};
static_assert(sizeof(sFileIdentifier20) == 12 * sizeof(uint8_t));
// «KTX 11»\r\n\x1A\n
const StaticArray<uint8_t, 12> sFileIdentifier10{
    {0xAB, 0x4B, 0x54, 0x58, 0x20, 0x31, 0x31, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A}};
static_assert(sizeof(sFileIdentifier10) == sizeof(sFileIdentifier20));

struct Ktx10Header
{
    uint32_t endianness{0};
    uint32_t glType{0};
    uint32_t glTypeSize{0};
    uint32_t glFormat{0};
    uint32_t glInternalFormat{0};
    uint32_t glBaseInternalFormat{0};
    uint32_t pixelWidth{0};
    uint32_t pixelHeight{0};
    uint32_t pixelDepth{0};
    uint32_t numberOfArrayElements{0};
    uint32_t numberOfFaces{0};
    uint32_t numberOfMipmapLevels{0};
    uint32_t bytesOfKeyValueData{0};
};

} // namespace

Ktx readKtx(Allocator &alloc, const std::filesystem::path &path)
{
    std::ifstream inFile{path, std::ios_base::binary};

    StaticArray<uint8_t, 12> identifier;
    static_assert(sizeof(identifier) == sizeof(sFileIdentifier20));
    readRawSpan(inFile, identifier.mut_span());

    if (memcmp(
            identifier.data(), sFileIdentifier20.data(),
            identifier.size() * sizeof(uint8_t)) == 0)
        throw std::runtime_error("KTX 2.0 is not supported");
    WHEELS_ASSERT(
        memcmp(
            identifier.data(), sFileIdentifier10.data(),
            identifier.size() * sizeof(uint8_t)) == 0 &&
        "File doesn't appear to be a KTX");

    Ktx10Header header;
    readRaw(inFile, header);

    WHEELS_ASSERT(
        header.endianness == 0x0403'0201 &&
        "KTX and program endianness don't match");

    // GL_HALF_FLOAT, GL_RGBA(16F)
    WHEELS_ASSERT(
        header.glType == 0x140b && header.glFormat == 0x1908 &&
        header.glInternalFormat == 0x881a &&
        header.glBaseInternalFormat == header.glFormat &&
        "Only RGBA16F is supported");
    const vk::Format format = vk::Format::eR16G16B16A16Sfloat;
    const uint32_t blockWidth = 1;
    const uint32_t blockHeight = 1;
    const uint32_t blockDepth = 1;
    const uint32_t blockByteCount = 8;

    // Ignore key value data
    inFile.seekg(header.bytesOfKeyValueData, std::ios_base::cur);

    Ktx ret{
        .width = header.pixelWidth,
        .height = std::max(header.pixelHeight, 1u),
        .depth = std::max(header.pixelDepth, 1u),
        .format = format,
        .arrayLayerCount = std::max(header.numberOfArrayElements, 1u),
        .faceCount = std::max(header.numberOfFaces, 1u),
        .mipLevelCount = std::max(1u, header.numberOfMipmapLevels),
        .data = Array<uint8_t>{alloc},
        .levelByteOffsets = Array<uint32_t>{alloc},
    };
    WHEELS_ASSERT(ret.width > 0);

    const bool isCubemap =
        header.numberOfArrayElements == 1 || header.numberOfFaces == 6;

    // Let's reserve conservatively to avoid reallocations. Each mip his half of
    // the previous one so 3x the size of mip0 layers and faces is more than
    // enough even for block compressed and weirdly rounded data.
    const uint32_t conservativeByteCount =
        3 * ret.arrayLayerCount * ret.faceCount *
        roundedUpQuotient(ret.width, blockWidth) *
        roundedUpQuotient(ret.height, blockHeight) *
        roundedUpQuotient(ret.depth, blockDepth) * blockByteCount;
    ret.data.reserve(conservativeByteCount);
    ret.levelByteOffsets.reserve(
        asserted_cast<size_t>(ret.mipLevelCount) *
        asserted_cast<size_t>(ret.arrayLayerCount) *
        asserted_cast<size_t>(ret.faceCount));

    // Levels are stored mips[layers[faces[z_slices[rows[pixels/blocks[]]]]]]
    for (uint32_t iMip = 0; iMip < ret.mipLevelCount; ++iMip)
    {
        uint32_t imageSize = 0;
        readRaw(inFile, imageSize);

        uint32_t cubePadding = 0;
        if (isCubemap)
        {
            cubePadding = imageSize % 4;
            // Cubemap imageSize is the size of one face
            imageSize *= 6;
        }
        WHEELS_ASSERT(
            cubePadding == 0 && "Parsing expects tightly packed faces");

        const uint32_t mipStartOffset =
            asserted_cast<uint32_t>(ret.data.size());
        ret.data.resize(ret.data.size() + imageSize);
        // We checked that cubes have no padding so we can just read the
        // whole size in one go
        readRawSpan(inFile, Span{ret.data.data() + mipStartOffset, imageSize});

        // Figure out layer/face offsets separately
        uint32_t readMipBytes = 0;
        const uint32_t faceByteCount = imageSize / ret.faceCount;
        for (uint32_t iLayer = 0; iLayer < ret.arrayLayerCount; ++iLayer)
        {
            for (uint32_t iFace = 0; iFace < ret.faceCount; ++iFace)
            {
                const uint32_t faceStartOffset = mipStartOffset + readMipBytes;
                ret.levelByteOffsets.push_back(faceStartOffset);

                readMipBytes += faceByteCount;
                WHEELS_ASSERT(readMipBytes <= imageSize);

                // We already checked that there is no cube padding
            }
        }
        WHEELS_ASSERT(readMipBytes == imageSize);

        const uint32_t mipPadding = readMipBytes % 4;
        inFile.seekg(mipPadding, std::ios_base::cur);
    }

    return ret;
}
