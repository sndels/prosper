#include "Mesh.hpp"


Mesh::Mesh(const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices, Device* device) :
    _device(device),
    _indexCount(indices.size())
{
    createVertexBuffer(vertices);
    createIndexBuffer(indices);
}

Mesh::~Mesh()
{
    if (_device != nullptr) {
        _device->logical().destroyBuffer(_indexBuffer.handle);
        _device->logical().freeMemory(_indexBuffer.memory);

        _device->logical().destroyBuffer(_vertexBuffer.handle);
        _device->logical().freeMemory(_vertexBuffer.memory);
    }
}

Mesh::Mesh(Mesh&& other) :
    _device(other._device),
    _vertexBuffer(other._vertexBuffer),
    _indexBuffer(other._indexBuffer),
    _indexCount(other._indexCount)
{
    other._device = nullptr;
    other._vertexBuffer = {{}, {}};
    other._indexBuffer = {{}, {}};
    other._indexCount = 0;
}

void Mesh::draw(vk::CommandBuffer commandBuffer)
{
    // Bind
    const vk::Buffer vertexBuffers[] = {_vertexBuffer.handle};
    const vk::DeviceSize offsets[] = {0};
    commandBuffer.bindVertexBuffers(0, 1, vertexBuffers, offsets);
    commandBuffer.bindIndexBuffer(_indexBuffer.handle, 0, vk::IndexType::eUint32);

    // Draw
    commandBuffer.drawIndexed(_indexCount, 1, 0, 0, 0);
}

void Mesh::createVertexBuffer(const std::vector<Vertex>& vertices)
{
    vk::DeviceSize bufferSize = sizeof(Vertex) * vertices.size();

    // Create staging buffer
    Buffer stagingBuffer = _device->createBuffer(
        bufferSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible |
        vk::MemoryPropertyFlagBits::eHostCoherent
    );

    // Move vertex data to it
    void* data;
    _device->logical().mapMemory(stagingBuffer.memory, 0, bufferSize, {}, &data);
    memcpy(data, vertices.data(), static_cast<size_t>(bufferSize));
    _device->logical().unmapMemory(stagingBuffer.memory);

    // Create vertex buffer
    _vertexBuffer = _device->createBuffer(
        bufferSize,
        vk::BufferUsageFlagBits::eVertexBuffer |
        vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal
    );
    _device->copyBuffer(stagingBuffer, _vertexBuffer, bufferSize);

    // Clean up
    _device->logical().destroyBuffer(stagingBuffer.handle);
    _device->logical().freeMemory(stagingBuffer.memory);
}

void Mesh::createIndexBuffer(const std::vector<uint32_t>& indices)
{
    vk::DeviceSize bufferSize = sizeof(uint32_t) * indices.size();

    // Create staging buffer
    Buffer stagingBuffer = _device->createBuffer(
        bufferSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible |
        vk::MemoryPropertyFlagBits::eHostCoherent
    );

    // Move index data to it
    void* data;
    _device->logical().mapMemory(stagingBuffer.memory, 0, bufferSize, {}, &data);
    memcpy(data, indices.data(), static_cast<size_t>(bufferSize));
    _device->logical().unmapMemory(stagingBuffer.memory);

    // Create index buffer
    _indexBuffer = _device->createBuffer(
        bufferSize,
        vk::BufferUsageFlagBits::eIndexBuffer |
        vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal
    );
    _device->copyBuffer(stagingBuffer, _indexBuffer, bufferSize);

    // Clean up
    _device->logical().destroyBuffer(stagingBuffer.handle, nullptr);
    _device->logical().freeMemory(stagingBuffer.memory, nullptr);
}
