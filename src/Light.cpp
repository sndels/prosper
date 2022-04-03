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
    const auto &buffer = this->uniformBuffers[nextImage];
    void *mapped = device->map(buffer);
    memcpy(mapped, &this->parameters, sizeof(Parameters));
    device->unmap(buffer);
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
    const auto &buffer = this->storageBuffers[nextImage];
    void *mapped = device->map(buffer);
    memcpy(mapped, &this->bufferData, sizeof(PointLights::BufferData));
    device->unmap(buffer);
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
    const auto &buffer = this->storageBuffers[nextImage];
    void *mapped = device->map(buffer);
    memcpy(mapped, &this->bufferData, sizeof(SpotLights::BufferData));
    device->unmap(buffer);
}
