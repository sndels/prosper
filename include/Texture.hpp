#ifndef PROSPER_TEXTURE_HPP
#define PROSPER_TEXTURE_HPP

#include "Device.hpp"

#include <gli/gli.hpp>

#include <filesystem>

namespace tinygltf
{
struct Image;
struct Sampler;
}; // namespace tinygltf

class Texture
{
  public:
    Texture(Device *device);
    virtual ~Texture();

    Texture(const Texture &other) = delete;
    Texture(Texture &&other) noexcept;
    Texture &operator=(const Texture &other) = delete;
    Texture &operator=(Texture &&other) noexcept;

    [[nodiscard]] virtual vk::DescriptorImageInfo imageInfo() const = 0;

  protected:
    void destroy();

    // Texture with null device is invalid or moved
    Device *_device{nullptr};
    Image _image;
};

class Texture2D : public Texture
{
  public:
    Texture2D(Device *device, const std::filesystem::path &path, bool mipmap);
    Texture2D(Device *device, const tinygltf::Image &image, bool mipmap);

    [[nodiscard]] virtual vk::DescriptorImageInfo imageInfo() const override;

  private:
    [[nodiscard]] Buffer stagePixels(
        const uint8_t *pixels, const vk::Extent2D &extent) const;
    void createImage(const Buffer &stagingBuffer, const ImageCreateInfo &info);
    void createMipmaps(const vk::Extent2D &extent, uint32_t mipLevels);
};

class TextureCubemap : public Texture
{
  public:
    TextureCubemap(Device *device, const std::filesystem::path &path);
    ~TextureCubemap();

    [[nodiscard]] virtual vk::DescriptorImageInfo imageInfo() const override;

  private:
    void copyPixels(
        const gli::texture_cube &cube,
        const vk::ImageSubresourceRange &subresourceRange) const;

    vk::Sampler _sampler;
};

#endif // PROSPER_TEXTURE_HPP
