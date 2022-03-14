#include "Light.hpp"

[[nodiscard]] std::vector<vk::DescriptorBufferInfo> DirectionalLight::
    bufferInfos() const
{
    std::vector<vk::DescriptorBufferInfo> infos;
    for (const auto &buffer : this->uniformBuffers)
        infos.push_back(vk::DescriptorBufferInfo{
            .buffer = buffer.handle,
            .offset = 0,
            .range = sizeof(Parameters),
        });

    return infos;
}

void DirectionalLight::updateBuffer(
    const Device *device, const uint32_t nextImage) const
{
    void *data = nullptr;
    device->map(this->uniformBuffers[nextImage].allocation, &data);
    memcpy(data, &this->parameters, sizeof(Parameters));
    device->unmap(this->uniformBuffers[nextImage].allocation);
}

[[nodiscard]] std::vector<vk::DescriptorBufferInfo> PointLights::bufferInfos()
    const
{
    std::vector<vk::DescriptorBufferInfo> infos;
    for (const auto &buffer : this->storageBuffers)
        infos.push_back(vk::DescriptorBufferInfo{
            .buffer = buffer.handle,
            .offset = 0,
            .range = sizeof(PointLights::BufferData),
        });

    return infos;
}

void PointLights::updateBuffer(
    const Device *device, const uint32_t nextImage) const
{
    void *data = nullptr;
    device->map(this->storageBuffers[nextImage].allocation, &data);
    memcpy(data, &this->bufferData, sizeof(PointLights::BufferData));
    device->unmap(this->storageBuffers[nextImage].allocation);
}

[[nodiscard]] std::vector<vk::DescriptorBufferInfo> SpotLights::bufferInfos()
    const
{
    std::vector<vk::DescriptorBufferInfo> infos;
    for (const auto &buffer : this->storageBuffers)
        infos.push_back(vk::DescriptorBufferInfo{
            .buffer = buffer.handle,
            .offset = 0,
            .range = sizeof(SpotLights::BufferData),
        });

    return infos;
}

void SpotLights::updateBuffer(
    const Device *device, const uint32_t nextImage) const
{
    void *data = nullptr;
    device->map(this->storageBuffers[nextImage].allocation, &data);
    memcpy(data, &this->bufferData, sizeof(SpotLights::BufferData));
    device->unmap(this->storageBuffers[nextImage].allocation);
}
