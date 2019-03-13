#include "Mesh.hpp"


Mesh::Mesh(const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices, Device* device) :
    _device(device),
    _indexCount(static_cast<uint32_t>(indices.size()))
{
    createVertexBuffer(vertices);
    createIndexBuffer(indices);
}

Mesh::~Mesh()
{
    if (_device != nullptr) {
        vkDestroyBuffer(_device->handle(), _indexBuffer.handle, nullptr);
        vkFreeMemory(_device->handle(), _indexBuffer.memory, nullptr);

        vkDestroyBuffer(_device->handle(), _vertexBuffer.handle, nullptr);
        vkFreeMemory(_device->handle(), _vertexBuffer.memory, nullptr);
    }
}

Mesh::Mesh(Mesh&& other) :
    _device(other._device),
    _vertexBuffer(other._vertexBuffer),
    _indexBuffer(other._indexBuffer),
    _indexCount(other._indexCount)
{
    other._device = nullptr;
    other._vertexBuffer = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    other._indexBuffer = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    other._indexCount = 0;
}

void Mesh::draw(VkCommandBuffer commandBuffer)
{
    // Bind
    VkBuffer vertexBuffers[] = {_vertexBuffer.handle};
    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
    vkCmdBindIndexBuffer(commandBuffer, _indexBuffer.handle, 0, VK_INDEX_TYPE_UINT32);

    // Draw
    vkCmdDrawIndexed(commandBuffer, _indexCount, 1, 0, 0, 0);
}

void Mesh::createVertexBuffer(const std::vector<Vertex>& vertices)
{
    VkDeviceSize bufferSize = sizeof(Vertex) * vertices.size();

    // Create staging buffer
    Buffer stagingBuffer;
    _device->createBuffer(&stagingBuffer, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    // Move vertex data to it
    void* data;
    vkMapMemory(_device->handle(), stagingBuffer.memory, 0, bufferSize, 0, &data);
    memcpy(data, vertices.data(), static_cast<size_t>(bufferSize));
    vkUnmapMemory(_device->handle(), stagingBuffer.memory);

    // Create vertex buffer
    _device->createBuffer(&_vertexBuffer, bufferSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    _device->copyBuffer(stagingBuffer, _vertexBuffer, bufferSize);

    // Clean up
    vkDestroyBuffer(_device->handle(), stagingBuffer.handle, nullptr);
    vkFreeMemory(_device->handle(), stagingBuffer.memory, nullptr);
}

void Mesh::createIndexBuffer(const std::vector<uint32_t>& indices)
{
    VkDeviceSize bufferSize = sizeof(uint32_t) * indices.size();

    // Create staging buffer
    Buffer stagingBuffer;
    _device->createBuffer(&stagingBuffer, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    // Move index data to it
    void* data;
    vkMapMemory(_device->handle(), stagingBuffer.memory, 0, bufferSize, 0, &data);
    memcpy(data, indices.data(), static_cast<size_t>(bufferSize));
    vkUnmapMemory(_device->handle(), stagingBuffer.memory);

    // Create index buffer
    _device->createBuffer(&_indexBuffer, bufferSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    _device->copyBuffer(stagingBuffer, _indexBuffer, bufferSize);

    // Clean up
    vkDestroyBuffer(_device->handle(), stagingBuffer.handle, nullptr);
    vkFreeMemory(_device->handle(), stagingBuffer.memory, nullptr);
}
