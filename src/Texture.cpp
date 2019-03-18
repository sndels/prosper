#include "Texture.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

Texture::Texture(Device* device, const std::string& path) :
    _device(device)
{
    const auto [pixels, extent] = loadFromFile(path);
    const auto stagingBuffer = stagePixels(pixels, extent);

    const vk::ImageSubresourceRange subresourceRange{
            vk::ImageAspectFlagBits::eColor,
            0, // baseMipLevel
            1, // levelCount
            0, // baseArrayLayer
            1 // layerCount
    };

    createImage(stagingBuffer, extent, subresourceRange);
    createImageView(subresourceRange);
    createSampler();

    stbi_image_free(pixels);
    _device->logical().destroy(stagingBuffer.handle);
    _device->logical().free(stagingBuffer.memory);
}

Texture::~Texture()
{
    _device->logical().destroy(_imageView);
    _device->logical().destroy(_sampler);
    _device->logical().destroy(_image.handle);
    _device->logical().free(_image.memory);
}

vk::DescriptorImageInfo Texture::imageInfo() const
{
    return vk::DescriptorImageInfo{
        _sampler,
        _imageView,
        vk::ImageLayout::eShaderReadOnlyOptimal
    };
}

std::pair<uint8_t*, vk::Extent2D> Texture::loadFromFile(const std::string& path)
{
    // Load from file
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

Buffer Texture::stagePixels(const uint8_t* pixels, const vk::Extent2D extent)
{
    // Upload to staging buffer
    const vk::DeviceSize imageSize = extent.width * extent.height * 4;
    const Buffer buffer = _device->createBuffer(
        imageSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible |
        vk::MemoryPropertyFlagBits::eHostCoherent
    );

    void* data;
    _device->logical().mapMemory(buffer.memory, 0, imageSize, {}, &data);
    memcpy(data, pixels, static_cast<size_t>(imageSize));
    _device->logical().unmapMemory(buffer.memory);

    return buffer;
}

void Texture::createImage(const Buffer& stagingBuffer, const vk::Extent2D extent, const vk::ImageSubresourceRange& subresourceRange)
{
    _image = _device->createImage(
        extent,
        vk::Format::eR8G8B8A8Unorm,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eTransferDst |
        vk::ImageUsageFlagBits::eSampled,
        vk::MemoryPropertyFlagBits::eDeviceLocal
    );

    _device->transitionImageLayout(
        _image,
        vk::Format::eR8G8B8A8Unorm,
        subresourceRange,
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eTransferDstOptimal
    );
    _device->copyBufferToImage(stagingBuffer, _image, extent);
    _device->transitionImageLayout(
        _image,
        vk::Format::eR8G8B8A8Unorm,
        subresourceRange,
        vk::ImageLayout::eTransferDstOptimal,
        vk::ImageLayout::eShaderReadOnlyOptimal
    );
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
}

void Texture::createSampler()
{
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
        16 // maxAnisotropy
    });
}
