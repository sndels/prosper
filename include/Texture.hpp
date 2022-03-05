#ifndef PROSPER_TEXTURE_HPP
#define PROSPER_TEXTURE_HPP

// CMake doesn't seem to support MSVC /external -stuff yet
#ifdef _MSC_VER
#pragma warning(push, 0)
#endif // _MSC_VER

#include <gli/gli.hpp>

namespace tinygltf
{
struct Image;
struct Sampler;
}; // namespace tinygltf

#ifdef _MSC_VER
#pragma warning(pop)
#endif // _MSC_VER

#include "Device.hpp"

#include <filesystem>

class Texture
{
  public:
    Texture(Device *device);
    ~Texture();

    Texture(const Texture &other) = delete;
    Texture(Texture &&other) noexcept;
    Texture &operator=(const Texture &other) = delete;
    Texture &operator=(Texture &&other) noexcept;

    [[nodiscard]] vk::DescriptorImageInfo imageInfo() const;

  protected:
    void destroy();

    // Texture with null device is invalid or moved
    Device *_device{nullptr};
    Image _image;
    vk::Sampler _sampler;
};

class Texture2D : public Texture
{
  public:
    Texture2D(Device *device, const std::filesystem::path &path, bool mipmap);
    Texture2D(
        Device *device, const tinygltf::Image &image,
        const tinygltf::Sampler &sampler, bool mipmap);

  private:
    [[nodiscard]] Buffer stagePixels(
        const uint8_t *pixels, const vk::Extent2D &extent) const;
    void createImage(
        const Buffer &stagingBuffer, const vk::Extent2D &extent,
        const vk::ImageSubresourceRange &subresourceRange);
    void createMipmaps(const vk::Extent2D &extent, uint32_t mipLevels);
    void createSampler(uint32_t mipLevels);
    void createSampler(const tinygltf::Sampler &sampler, uint32_t mipLevels);
};

class TextureCubemap : public Texture
{
  public:
    TextureCubemap(Device *device, const std::filesystem::path &path);

  private:
    void copyPixels(
        const gli::texture_cube &cube,
        const vk::ImageSubresourceRange &subresourceRange) const;
};

#endif // PROSPER_TEXTURE_HPP
