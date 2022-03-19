#include "Mesh.hpp"

#include "Utils.hpp"

Mesh::Mesh(
    Device *device, const std::vector<Vertex> &vertices,
    const std::vector<uint32_t> &indices)
: _device{device}
, _vertexCount{asserted_cast<uint32_t>(vertices.size())}
, _indexCount{asserted_cast<uint32_t>(indices.size())}
{
    createVertexBuffer(vertices);
    createIndexBuffer(indices);
}

Mesh::~Mesh() { destroy(); }

Mesh::Mesh(Mesh &&other) noexcept
: _device{other._device}
, _vertexBuffer{other._vertexBuffer}
, _indexBuffer{other._indexBuffer}
, _vertexCount{other._vertexCount}
, _indexCount{other._indexCount}
{
    other._device = nullptr;
}

Mesh &Mesh::operator=(Mesh &&other) noexcept
{
    if (this != &other)
    {
        destroy();
        _device = other._device;
        _vertexBuffer = other._vertexBuffer;
        _indexBuffer = other._indexBuffer;
        _vertexCount = other._vertexCount;
        _indexCount = other._indexCount;

        other._device = nullptr;
    }

    return *this;
}

vk::Buffer Mesh::vertexBuffer() const { return _vertexBuffer.handle; }
vk::Buffer Mesh::indexBuffer() const { return _indexBuffer.handle; }
uint32_t Mesh::vertexCount() const { return _vertexCount; }
uint32_t Mesh::indexCount() const { return _indexCount; }

void Mesh::draw(vk::CommandBuffer commandBuffer) const
{
    const vk::DeviceSize offset = 0;
    commandBuffer.bindVertexBuffers(0, 1, &_vertexBuffer.handle, &offset);
    commandBuffer.bindIndexBuffer(
        _indexBuffer.handle, 0, vk::IndexType::eUint32);

    commandBuffer.drawIndexed(_indexCount, 1, 0, 0, 0);
}

void Mesh::destroy()
{
    if (_device != nullptr)
    {
        _device->destroy(_indexBuffer);
        _device->destroy(_vertexBuffer);
    }
    _device = nullptr;
}

void Mesh::createVertexBuffer(const std::vector<Vertex> &vertices)
{
    const vk::DeviceSize bufferSize = sizeof(Vertex) * vertices.size();

    const Buffer stagingBuffer = _device->createBuffer(
        "MeshVertexStagingBuffer", bufferSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent,
        VMA_MEMORY_USAGE_CPU_TO_GPU);

    // Move vertex data to staging
    void *data = nullptr;
    _device->map(stagingBuffer.allocation, &data);
    memcpy(data, vertices.data(), asserted_cast<size_t>(bufferSize));
    _device->unmap(stagingBuffer.allocation);

    _vertexBuffer = _device->createBuffer(
        "MeshVertexBuffer", bufferSize,
        vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR |
            vk::BufferUsageFlagBits::eShaderDeviceAddress |
            vk::BufferUsageFlagBits::eVertexBuffer |
            vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal, VMA_MEMORY_USAGE_GPU_ONLY);

    const auto commandBuffer = _device->beginGraphicsCommands();

    const vk::BufferCopy copyRegion{
        .srcOffset = 0,
        .dstOffset = 0,
        .size = bufferSize,
    };
    commandBuffer.copyBuffer(
        stagingBuffer.handle, _vertexBuffer.handle, 1, &copyRegion);

    _device->endGraphicsCommands(commandBuffer);

    _device->destroy(stagingBuffer);
}

void Mesh::createIndexBuffer(const std::vector<uint32_t> &indices)
{
    const vk::DeviceSize bufferSize = sizeof(uint32_t) * indices.size();

    const Buffer stagingBuffer = _device->createBuffer(
        "MeshIndexStagingBuffer", bufferSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent,
        VMA_MEMORY_USAGE_CPU_TO_GPU);

    // Move index data to staging
    void *data = nullptr;
    _device->map(stagingBuffer.allocation, &data);
    memcpy(data, indices.data(), asserted_cast<size_t>(bufferSize));
    _device->unmap(stagingBuffer.allocation);

    _indexBuffer = _device->createBuffer(
        "MeshIndexBuffer", bufferSize,
        vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR |
            vk::BufferUsageFlagBits::eShaderDeviceAddress |
            vk::BufferUsageFlagBits::eIndexBuffer |
            vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal, VMA_MEMORY_USAGE_GPU_ONLY);

    const auto commandBuffer = _device->beginGraphicsCommands();

    const vk::BufferCopy copyRegion{
        .srcOffset = 0,
        .dstOffset = 0,
        .size = bufferSize,
    };
    commandBuffer.copyBuffer(
        stagingBuffer.handle, _indexBuffer.handle, 1, &copyRegion);

    _device->endGraphicsCommands(commandBuffer);

    _device->destroy(stagingBuffer);
}
