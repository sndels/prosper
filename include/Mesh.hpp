#ifndef PROSPER_MESH_HPP
#define PROSPER_MESH_HPP

#include <vector>

#include "Device.hpp"
#include "Material.hpp"
#include "Vertex.hpp"

class Mesh {
public:
    Mesh(const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices, Material* material, Device* device);
    ~Mesh();

    Mesh(const Mesh& other) = delete;
    Mesh(Mesh&& other);
    Mesh operator=(const Mesh& other) = delete;

    const Material& material() const;

    void draw(vk::CommandBuffer commandBuffer) const;

private:
    void createVertexBuffer(const std::vector<Vertex>& vertices);
    void createIndexBuffer(const std::vector<uint32_t>& indices);

    Device* _device;
    Material* _material;
    Buffer _vertexBuffer;
    Buffer _indexBuffer;
    uint32_t _indexCount;

};

#endif // PROSPER_MESH_HPP
