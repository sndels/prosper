#ifndef PROSPER_MESH_HPP
#define PROSPER_MESH_HPP

#include <vector>

#include "Device.hpp"
#include "Material.hpp"
#include "Vertex.hpp"

class Mesh
{
  public:
    Mesh(
        Device *device, const std::vector<Vertex> &vertices,
        const std::vector<uint32_t> &indices);
    ~Mesh();

    Mesh(const Mesh &other) = delete;
    Mesh(Mesh &&other) noexcept;
    Mesh &operator=(const Mesh &other) = delete;
    Mesh &operator=(Mesh &&other) noexcept;

    [[nodiscard]] vk::Buffer vertexBuffer() const;
    [[nodiscard]] vk::Buffer indexBuffer() const;
    [[nodiscard]] uint32_t vertexCount() const;
    [[nodiscard]] uint32_t indexCount() const;

    void draw(vk::CommandBuffer commandBuffer) const;

  private:
    void destroy();

    Device *_device{nullptr};
    Buffer _vertexBuffer;
    Buffer _indexBuffer;
    uint32_t _vertexCount{0};
    uint32_t _indexCount{0};
};

#endif // PROSPER_MESH_HPP
