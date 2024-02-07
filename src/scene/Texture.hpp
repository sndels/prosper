#ifndef PROSPER_SCENE_TEXTURE_HPP
#define PROSPER_SCENE_TEXTURE_HPP

#include "../gfx/Fwd.hpp"
#include "../gfx/Resources.hpp"

#include <wheels/allocators/scoped_scratch.hpp>

#include <filesystem>

namespace gli
{
class texture_cube;
}

namespace tinygltf
{
struct Image;
}; // namespace tinygltf

class Texture
{
  public:
    Texture() noexcept = default;
    virtual ~Texture();

    Texture(const Texture &other) = delete;
    Texture(Texture &&other) noexcept;
    Texture &operator=(const Texture &other) = delete;
    Texture &operator=(Texture &&other) noexcept;

    [[nodiscard]] virtual vk::DescriptorImageInfo imageInfo() const = 0;
    [[nodiscard]] virtual vk::Image nativeHandle() const;

  protected:
    void init(Device *device);

    void destroy();

    // Texture with null device is invalid or moved
    Device *_device{nullptr};
    Image _image;
};

class Texture2D : public Texture
{
  public:
    // The image is ready and stagingBuffer can be freed once cb is submitted
    // and has finished executing.
    void init(
        wheels::ScopedScratch scopeAlloc, Device *device,
        const std::filesystem::path &path, vk::CommandBuffer cb,
        const Buffer &stagingBuffer, bool mipmap,
        ImageState initialState = ImageState::Unknown);

    [[nodiscard]] vk::DescriptorImageInfo imageInfo() const override;
};

class TextureCubemap : public Texture
{
  public:
    TextureCubemap() noexcept = default;
    ~TextureCubemap() override;

    TextureCubemap(const TextureCubemap &other) = delete;
    TextureCubemap(TextureCubemap &&other) noexcept;
    TextureCubemap &operator=(const TextureCubemap &other) = delete;
    TextureCubemap &operator=(TextureCubemap &&other) noexcept;

    void init(
        wheels::ScopedScratch scopeAlloc, Device *device,
        const std::filesystem::path &path);

    [[nodiscard]] vk::DescriptorImageInfo imageInfo() const override;

  private:
    void copyPixels(
        wheels::ScopedScratch scopeAlloc, const gli::texture_cube &cube,
        const vk::ImageSubresourceRange &subresourceRange) const;

    vk::Sampler _sampler;
};

#endif // PROSPER_SCENE_TEXTURE_HPP
