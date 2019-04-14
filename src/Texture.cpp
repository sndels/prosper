#include "Texture.hpp"

#include <cmath>
#include <iostream>

#include <stb_image.h>

#if defined(_WIN32) or defined(_WIN64)
    // Windows' header doesn't include these
    #define GL_CLAMP_TO_EDGE 0x812F
    #define GL_MIRRORED_REPEAT 0x8370
#endif // _WIN32 or _WIN64

namespace {
    vk::Filter getVkFilterMode(int glEnum)
    {
        switch (glEnum) {
        case GL_NEAREST:
            return vk::Filter::eNearest;
        case GL_LINEAR:
            return vk::Filter::eLinear;
        case GL_NEAREST_MIPMAP_NEAREST:
            return vk::Filter::eNearest;
        case GL_NEAREST_MIPMAP_LINEAR:
            return vk::Filter::eNearest;
        case GL_LINEAR_MIPMAP_NEAREST:
            return vk::Filter::eLinear;
        case GL_LINEAR_MIPMAP_LINEAR:
            return vk::Filter::eLinear;
        }

        std::cerr << "Invalid gl filter " << glEnum << std::endl;
        return vk::Filter::eLinear;
    }

    vk::SamplerAddressMode getVkAddressMode(int glEnum)
    {
        switch (glEnum) {
        case GL_CLAMP_TO_EDGE:
            return vk::SamplerAddressMode::eClampToEdge;
        case GL_MIRRORED_REPEAT:
            return vk::SamplerAddressMode::eMirroredRepeat;
        case GL_REPEAT:
            return vk::SamplerAddressMode::eRepeat;
        }
        std::cerr << "Invalid gl wrapping mode " << glEnum << std::endl;
        return vk::SamplerAddressMode::eClampToEdge;
    }

    std::pair<uint8_t*, vk::Extent2D> pixelsFromFile(const std::string& path)
    {
        int w, h, channels;
        stbi_uc* pixels = stbi_load(path.c_str(), &w, &h, &channels, STBI_rgb_alpha);
        if (pixels == nullptr)
            throw std::runtime_error("Failed to load texture '" + path + "'");

        return std::pair{
            pixels,
            vk::Extent2D{
                static_cast<uint32_t>(w),
                static_cast<uint32_t>(h)
            }
        };
    }
}

Texture::Texture(Device* device, const std::string& path, const bool mipmap) :
    _device(device)
{
    const auto [pixels, extent] = pixelsFromFile(path);
    const auto stagingBuffer = stagePixels(pixels, extent);

    const uint32_t mipLevels = mipmap ?
        static_cast<uint32_t>(floor(log2(std::max(extent.width, extent.height)))) + 1 :
        1;
    const vk::ImageSubresourceRange subresourceRange{
            vk::ImageAspectFlagBits::eColor,
            0, // baseMipLevel
            mipLevels, // levelCount
            0, // baseArrayLayer
            1 // layerCount
    };

    createImage(stagingBuffer, extent, subresourceRange);
    createImageView(subresourceRange);
    createSampler(mipLevels);

    stbi_image_free(pixels);
    _device->destroy(stagingBuffer);
}

Texture::Texture(Device* device, const tinygltf::Image& image, const tinygltf::Sampler& sampler, const bool mipmap) :
    _device(device)
{
    // TODO: support
    if (image.pixel_type != TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE)
        throw std::runtime_error("Unsupported glTF pixel_type");

    const uint8_t* pixels;
    std::vector<uint8_t> tmpPixels;
    if (image.component < 3)
        throw std::runtime_error("Image with less than 3 components");
    else if (image.component == 3) {
        std::cerr << "3 component texture" << std::endl;
        // Add fourth channel
        // TODO: Do only if rgb-textures are unsupported
        tmpPixels.resize(image.width * image.height * 4);
        const auto* rgb = image.image.data();
        auto* rgba = tmpPixels.data();
        for (int i = 0; i < image.width * image.height; ++i) {
            rgba[0] = rgb[0];
            rgba[1] = rgb[1];
            rgba[2] = rgb[2];
            rgb += 3;
            rgba += 4;
        }
        pixels = tmpPixels.data();
    } else
        pixels = reinterpret_cast<const uint8_t*>(image.image.data());
    const VkExtent2D extent{
        static_cast<uint32_t>(image.width),
        static_cast<uint32_t>(image.height)
    };
    const auto stagingBuffer = stagePixels(pixels, extent);

    const uint32_t mipLevels = mipmap ?
        static_cast<uint32_t>(floor(log2(std::max(extent.width, extent.height)))) + 1 :
        1;
    const vk::ImageSubresourceRange subresourceRange{
            vk::ImageAspectFlagBits::eColor,
            0, // baseMipLevel
            mipLevels, // levelCount
            0, // baseArrayLayer
            1 // layerCount
    };

    createImage(stagingBuffer, extent, subresourceRange);
    createImageView(subresourceRange);
    createSampler(sampler, mipLevels);

    _device->destroy(stagingBuffer);
}

Texture::~Texture()
{
    _device->logical().destroy(_imageView);
    _device->logical().destroy(_sampler);
    _device->destroy(_image);
}

Texture::Texture(Texture&& other) :
    _device(other._device),
    _image(other._image),
    _imageView(other._imageView),
    _sampler(other._sampler)
{
    other._image.handle = nullptr;
    other._image.allocation = nullptr;
    other._imageView = nullptr;
    other._sampler = nullptr;
}

Texture& Texture::operator=(Texture&& other)
{
    if (this != &other) {
        _device = other._device;
        _image = other._image;
        _imageView = other._imageView;
        _sampler = other._sampler;
        other._image.handle = nullptr;
        other._image.allocation = nullptr;
        other._imageView = nullptr;
        other._sampler = nullptr;
    }
    return *this;
}

vk::DescriptorImageInfo Texture::imageInfo() const
{
    return vk::DescriptorImageInfo{
        _sampler,
        _imageView,
        vk::ImageLayout::eShaderReadOnlyOptimal
    };
}

Buffer Texture::stagePixels(const uint8_t* pixels, const vk::Extent2D extent)
{
    const vk::DeviceSize imageSize = extent.width * extent.height * 4;

    const Buffer stagingBuffer = _device->createBuffer(
        imageSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible |
        vk::MemoryPropertyFlagBits::eHostCoherent,
        VMA_MEMORY_USAGE_CPU_TO_GPU
    );

    void* data;
    _device->map(stagingBuffer.allocation, &data);
    memcpy(data, pixels, static_cast<size_t>(imageSize));
    _device->unmap(stagingBuffer.allocation);

    return stagingBuffer;
}

void Texture::createImage(const Buffer& stagingBuffer, const vk::Extent2D extent, const vk::ImageSubresourceRange& subresourceRange)
{
    // Both transfer source and destination as pixels will be transferred to it and
    // mipmaps will be generated from it
    _image = _device->createImage(
        extent,
        subresourceRange.levelCount,
        vk::Format::eR8G8B8A8Unorm,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eTransferSrc |
        vk::ImageUsageFlagBits::eTransferDst |
        vk::ImageUsageFlagBits::eSampled,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        VMA_MEMORY_USAGE_GPU_ONLY
    );

    _device->transitionImageLayout(
        _image,
        subresourceRange,
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eTransferDstOptimal
    );
    _device->copyBufferToImage(stagingBuffer, _image, extent);

    if (!_image.handle)
        std::cerr << "Null image" << std::endl;
    if (!_image.allocation)
        std::cerr << "Null image allocation" << std::endl;

    createMipmaps(extent, subresourceRange.levelCount);
}

void Texture::createMipmaps(const vk::Extent2D extent, const uint32_t mipLevels)
{
    // TODO: Check that the texture format supports linear filtering
    const auto buffer = _device->beginGraphicsCommands();

    vk::ImageMemoryBarrier barrier{
        vk::AccessFlagBits::eTransferWrite, // srcAccessMask
        vk::AccessFlagBits::eTransferRead, // dstAccessMask
        vk::ImageLayout::eTransferDstOptimal, // oldLayout
        vk::ImageLayout::eTransferSrcOptimal, // newLayout
        VK_QUEUE_FAMILY_IGNORED, // srcQueueFamilyIndex
        VK_QUEUE_FAMILY_IGNORED, // dstQueueFamilyIndex
        _image.handle,
        vk::ImageSubresourceRange{
            vk::ImageAspectFlagBits::eColor,
            0, // baseMipLevel
            1, // levelCount
            0, // baseArrayLayer
            1 // layerCount
        }
    };

    int32_t mipWidth = extent.width;
    int32_t mipHeight = extent.height;
    for (uint32_t i = 1; i < mipLevels; ++i) {
        // Make sure last operation finished and source is transitioned
        barrier.subresourceRange.baseMipLevel = i - 1;
        barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
        barrier.newLayout = vk::ImageLayout::eTransferSrcOptimal;
        barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;
        buffer.pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer, // srcStageMask
            vk::PipelineStageFlagBits::eTransfer, // dstStageMask
            {}, // dependencyFlags
            {}, nullptr, // memoryBarrierCount, ptr
            {}, nullptr, // bufferMemoryBarrierCount, ptr
            1, &barrier // imageMemoryBarrierCount, ptr
        );

        vk::ImageBlit blit{
            vk::ImageSubresourceLayers{
                vk::ImageAspectFlagBits::eColor,
                i - 1, // baseMipLevel
                0, // baseArrayLayer
                1 // layerCount
            }, // srcSubresource
            {{{0}, {mipWidth, mipHeight, 1}}}, // srtOffsets
            vk::ImageSubresourceLayers{
                vk::ImageAspectFlagBits::eColor,
                i, // baseMipLevel
                0, // baseArrayLayer
                1 // layerCount
            }, // dstSubresource
            {{
                {0},
                {
                    mipWidth > 1 ? mipWidth / 2 : 1,
                    mipHeight > 1 ? mipHeight / 2 : 1,
                    1
                }
            }} // srcOffsets
        };
        buffer.blitImage(
            _image.handle, // srcImage
            vk::ImageLayout::eTransferSrcOptimal, // srcImageLayout
            _image.handle, // dstImage
            vk::ImageLayout::eTransferDstOptimal, // dstImageLayout
            1, &blit, // regionCount, ptr
            vk::Filter::eLinear
        );

        // Source needs to be transitioned to shader read optimal
        barrier.oldLayout = vk::ImageLayout::eTransferSrcOptimal;
        barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        barrier.srcAccessMask = vk::AccessFlagBits::eTransferRead;
        barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
        buffer.pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer, // srcStageMask
            vk::PipelineStageFlagBits::eFragmentShader, // dstStageMask
            {}, // dependencyFlags
            {}, nullptr, // memoryBarrierCount, ptr
            {}, nullptr, // bufferMemoryBarrierCount, ptr
            1, &barrier // imageMemoryBarrierCount, ptr
        );

        if (mipWidth > 1)
            mipWidth /= 2;
        if (mipHeight > 1)
            mipHeight /= 2;
    }

    // Last mip level needs to be transitioned to shader read optimal
    barrier.subresourceRange.baseMipLevel = mipLevels - 1;
    barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
    barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
    barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
    buffer.pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer, // srcStageMask
        vk::PipelineStageFlagBits::eFragmentShader, // dstStageMask
        {}, // dependencyFlags
        {}, nullptr, // memoryBarrierCount, ptr
        {}, nullptr, // bufferMemoryBarrierCount, ptr
        1, &barrier // imageMemoryBarrierCount, ptr
    );

    _device->endGraphicsCommands(buffer);
}

void Texture::createImageView(const vk::ImageSubresourceRange& subresourceRange)
{
    _imageView = _device->logical().createImageView({
        {}, // flags
        _image.handle,
        vk::ImageViewType::e2D,
        vk::Format::eR8G8B8A8Unorm,
        vk::ComponentMapping{},
        subresourceRange
    });

    if (!_imageView)
        std::cerr << "Null image view" << std::endl;
}

void Texture::createSampler(const uint32_t mipLevels)
{
    // TODO: Use shared samplers
    _sampler = _device->logical().createSampler({
        {}, // flags
        vk::Filter::eLinear, // magFilter
        vk::Filter::eLinear, // minFilter
        vk::SamplerMipmapMode::eLinear,
        vk::SamplerAddressMode::eClampToEdge, // addressModeU
        vk::SamplerAddressMode::eClampToEdge, // addressModeV
        vk::SamplerAddressMode::eClampToEdge, // addressModeW
        0.f, // mipLodBias
        VK_TRUE, // anisotropyEnable
        16, // maxAnisotropy
        false, // compareEnable
        vk::CompareOp::eNever,
        0, // minLod
        static_cast<float>(mipLevels) // maxLod
    });
}

void Texture::createSampler(const tinygltf::Sampler& sampler, const uint32_t mipLevels)
{
    // TODO: Use shared samplers
    _sampler = _device->logical().createSampler({
        {}, // flags
        getVkFilterMode(sampler.magFilter),
        getVkFilterMode(sampler.minFilter),
        vk::SamplerMipmapMode::eLinear, // TODO
        getVkAddressMode(sampler.wrapS),
        getVkAddressMode(sampler.wrapT),
        vk::SamplerAddressMode::eClampToEdge, // addressModeW
        0.f, // mipLodBias
        VK_TRUE, // anisotropyEnable
        16, // maxAnisotropy
        false, // compareEnable
        vk::CompareOp::eNever,
        0, // minLod
        static_cast<float>(mipLevels) // maxLod
    });
}
