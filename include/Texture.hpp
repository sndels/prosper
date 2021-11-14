#ifndef PROSPER_TEXTURE_HPP
#define PROSPER_TEXTURE_HPP

#include <gli/gli.hpp>
#include <tiny_gltf.h>

#include "Device.hpp"

class Texture
{
  public:
    Texture(std::shared_ptr<Device> device);
    ~Texture();

    Texture(const Texture &other) = delete;
    Texture(Texture &&other);
    Texture &operator=(const Texture &other) = delete;
    Texture &operator=(Texture &&other);

    vk::DescriptorImageInfo imageInfo() const;

  protected:
    void destroy();

    // Texture with null device is invalid or moved
    std::shared_ptr<Device> _device = nullptr;
    Image _image;
    vk::Sampler _sampler;
};

class Texture2D : public Texture
{
  public:
    Texture2D(
        std::shared_ptr<Device> device, const std::string &path,
        const bool mipmap);
    Texture2D(
        std::shared_ptr<Device> device, const tinygltf::Image &image,
        const tinygltf::Sampler &sampler, const bool mipmap);

  private:
    Buffer stagePixels(const uint8_t *pixels, const vk::Extent2D extent) const;
    void createImage(
        const Buffer &stagingBuffer, const vk::Extent2D extent,
        const vk::ImageSubresourceRange &subresourceRange);
    void createMipmaps(
        const vk::Extent2D extent, const uint32_t mipLevels) const;
    void createSampler(const uint32_t mipLevels);
    void createSampler(
        const tinygltf::Sampler &sampler, const uint32_t mipLevels);
};

class TextureCubemap : public Texture
{
  public:
    TextureCubemap(std::shared_ptr<Device> device, const std::string &path);

  private:
    void copyPixels(
        const gli::texture_cube &cube,
        const vk::ImageSubresourceRange &subresourceRange) const;
};

#endif // PROSPER_TEXTURE_HPP
