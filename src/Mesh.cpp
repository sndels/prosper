#include "Mesh.hpp"

#include "Utils.hpp"

Mesh::Mesh(
    Device *device, const std::vector<Vertex> &vertices,
    const std::vector<uint32_t> &indices)
: _device{device}
, _vertexCount{asserted_cast<uint32_t>(vertices.size())}
, _indexCount{asserted_cast<uint32_t>(indices.size())}
{
    _vertexBuffer = _device->createBuffer(BufferCreateInfo{
        .byteSize = sizeof(Vertex) * vertices.size(),
        .usage = vk::BufferUsageFlagBits::
                     eAccelerationStructureBuildInputReadOnlyKHR |
                 vk::BufferUsageFlagBits::eShaderDeviceAddress |
                 vk::BufferUsageFlagBits::eVertexBuffer |
                 vk::BufferUsageFlagBits::eTransferDst,
        .properties = vk::MemoryPropertyFlagBits::eDeviceLocal,
        .initialData = vertices.data(),
        .debugName = "MeshVertexBuffer",
    });
    _indexBuffer = _device->createBuffer(BufferCreateInfo{
        .byteSize = sizeof(uint32_t) * indices.size(),
        .usage = vk::BufferUsageFlagBits::
                     eAccelerationStructureBuildInputReadOnlyKHR |
                 vk::BufferUsageFlagBits::eShaderDeviceAddress |
                 vk::BufferUsageFlagBits::eIndexBuffer |
                 vk::BufferUsageFlagBits::eTransferDst,
        .properties = vk::MemoryPropertyFlagBits::eDeviceLocal,
        .initialData = indices.data(),
        .debugName = "MeshIndexBuffer",
    });
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
