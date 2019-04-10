#include "Mesh.hpp"


Mesh::Mesh(const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices, Material* material, Device* device) :
    _device(device),
    _material(material),
    _indexCount(indices.size())
{
    createVertexBuffer(vertices);
    createIndexBuffer(indices);
}

Mesh::~Mesh()
{
    if (_device != nullptr) {
        _device->logical().destroy(_indexBuffer.handle);
        _device->logical().free(_indexBuffer.memory);

        _device->logical().destroy(_vertexBuffer.handle);
        _device->logical().free(_vertexBuffer.memory);
    }
}

Mesh::Mesh(Mesh&& other) :
    _device(other._device),
    _material(other._material),
    _vertexBuffer(other._vertexBuffer),
    _indexBuffer(other._indexBuffer),
    _indexCount(other._indexCount)
{
    other._material = nullptr;
    other._device = nullptr;
    other._vertexBuffer = {{}, {}};
    other._indexBuffer = {{}, {}};
    other._indexCount = 0;
}

const Material& Mesh::material() const
{
    return *_material;
}

void Mesh::draw(vk::CommandBuffer commandBuffer) const
{
    const vk::DeviceSize offset = 0;
    commandBuffer.bindVertexBuffers(0, 1, &_vertexBuffer.handle, &offset);
    commandBuffer.bindIndexBuffer(_indexBuffer.handle, 0, vk::IndexType::eUint32);

    commandBuffer.drawIndexed(_indexCount, 1, 0, 0, 0);
}

void Mesh::createVertexBuffer(const std::vector<Vertex>& vertices)
{
    const vk::DeviceSize bufferSize = sizeof(Vertex) * vertices.size();

    const Buffer stagingBuffer = _device->createBuffer(
        bufferSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible |
        vk::MemoryPropertyFlagBits::eHostCoherent
    );

    // Move vertex data to staging
    void* data;
    _device->logical().mapMemory(stagingBuffer.memory, 0, bufferSize, {}, &data);
    memcpy(data, vertices.data(), static_cast<size_t>(bufferSize));
    _device->logical().unmapMemory(stagingBuffer.memory);

    _vertexBuffer = _device->createBuffer(
        bufferSize,
        vk::BufferUsageFlagBits::eVertexBuffer |
        vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal
    );
    _device->copyBuffer(stagingBuffer, _vertexBuffer, bufferSize);

    _device->logical().destroy(stagingBuffer.handle);
    _device->logical().free(stagingBuffer.memory);
}

void Mesh::createIndexBuffer(const std::vector<uint32_t>& indices)
{
    const vk::DeviceSize bufferSize = sizeof(uint32_t) * indices.size();

    const Buffer stagingBuffer = _device->createBuffer(
        bufferSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible |
        vk::MemoryPropertyFlagBits::eHostCoherent
    );

    // Move index data to staging
    void* data;
    _device->logical().mapMemory(stagingBuffer.memory, 0, bufferSize, {}, &data);
    memcpy(data, indices.data(), static_cast<size_t>(bufferSize));
    _device->logical().unmapMemory(stagingBuffer.memory);

    _indexBuffer = _device->createBuffer(
        bufferSize,
        vk::BufferUsageFlagBits::eIndexBuffer |
        vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal
    );
    _device->copyBuffer(stagingBuffer, _indexBuffer, bufferSize);

    // Clean up
    _device->logical().destroy(stagingBuffer.handle);
    _device->logical().free(stagingBuffer.memory);
}
