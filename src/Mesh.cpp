#include "Mesh.hpp"

#include "VkUtils.hpp"

Mesh::Mesh(const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices, Material* material, std::shared_ptr<Device> device) :
    _device{device},
    _material{material},
    _indexCount{static_cast<uint32_t>(indices.size())}
{
    createVertexBuffer(vertices);
    createIndexBuffer(indices);
}

Mesh::~Mesh()
{
    if (_device) {
        _device->destroy(_indexBuffer);
        _device->destroy(_vertexBuffer);
    }
}

Mesh::Mesh(Mesh&& other) :
    _device{other._device},
    _material{other._material},
    _vertexBuffer{other._vertexBuffer},
    _indexBuffer{other._indexBuffer},
    _indexCount{other._indexCount}
{
    other._device = nullptr;
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
        vk::MemoryPropertyFlagBits::eHostCoherent,
        VMA_MEMORY_USAGE_CPU_TO_GPU
    );

    // Move vertex data to staging
    void* data;
    _device->map(stagingBuffer.allocation, &data);
    memcpy(data, vertices.data(), static_cast<size_t>(bufferSize));
    _device->unmap(stagingBuffer.allocation);

    _vertexBuffer = _device->createBuffer(
        bufferSize,
        vk::BufferUsageFlagBits::eVertexBuffer |
        vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        VMA_MEMORY_USAGE_GPU_ONLY
    );

    const auto commandBuffer = _device->beginGraphicsCommands();

    const vk::BufferCopy copyRegion{
        0, // srcOffset
        0, // dstOffset
        bufferSize
    };
    commandBuffer.copyBuffer(stagingBuffer.handle, _vertexBuffer.handle, 1, &copyRegion);

    _device->endGraphicsCommands(commandBuffer);

    _device->destroy(stagingBuffer);
}

void Mesh::createIndexBuffer(const std::vector<uint32_t>& indices)
{
    const vk::DeviceSize bufferSize = sizeof(uint32_t) * indices.size();

    const Buffer stagingBuffer = _device->createBuffer(
        bufferSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible |
        vk::MemoryPropertyFlagBits::eHostCoherent,
        VMA_MEMORY_USAGE_CPU_TO_GPU
    );

    // Move index data to staging
    void* data;
    _device->map(stagingBuffer.allocation, &data);
    memcpy(data, indices.data(), static_cast<size_t>(bufferSize));
    _device->unmap(stagingBuffer.allocation);

    _indexBuffer = _device->createBuffer(
        bufferSize,
        vk::BufferUsageFlagBits::eIndexBuffer |
        vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        VMA_MEMORY_USAGE_GPU_ONLY
    );

    const auto commandBuffer = _device->beginGraphicsCommands();

    const vk::BufferCopy copyRegion{
        0, // srcOffset
        0, // dstOffset
        bufferSize
    };
    commandBuffer.copyBuffer(stagingBuffer.handle, _indexBuffer.handle, 1, &copyRegion);

    _device->endGraphicsCommands(commandBuffer);

    _device->destroy(stagingBuffer);
}
