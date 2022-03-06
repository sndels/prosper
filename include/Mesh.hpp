#ifndef PROSPER_MESH_HPP
#define PROSPER_MESH_HPP

#include <vector>

#include "Device.hpp"
#include "Material.hpp"
#include "Vertex.hpp"

class Mesh
{
  public:
    struct PCBlock
    {
        uint32_t modelInstanceID{0};
        uint32_t materialID{0};
    };

    Mesh(
        const std::vector<Vertex> &vertices,
        const std::vector<uint32_t> &indices, uint32_t materialID,
        Device *device);
    ~Mesh();

    Mesh(const Mesh &other) = delete;
    Mesh(Mesh &&other) noexcept;
    Mesh &operator=(const Mesh &other) = delete;
    Mesh &operator=(Mesh &&other) noexcept;

    [[nodiscard]] uint32_t materialID() const;

    void draw(vk::CommandBuffer commandBuffer) const;

  private:
    void destroy();
    void createVertexBuffer(const std::vector<Vertex> &vertices);
    void createIndexBuffer(const std::vector<uint32_t> &indices);

    Device *_device{nullptr};
    uint32_t _materialID{0};
    Buffer _vertexBuffer;
    Buffer _indexBuffer;
    uint32_t _indexCount{0};
};

#endif // PROSPER_MESH_HPP
