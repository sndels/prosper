#include "Light.hpp"

#include "Utils.hpp"

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

void DirectionalLight::updateBuffer(const uint32_t nextImage) const
{
    memcpy(
        this->uniformBuffers[nextImage].mapped, &this->parameters,
        sizeof(Parameters));
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

void PointLights::updateBuffer(const uint32_t nextImage) const
{
    memcpy(
        this->storageBuffers[nextImage].mapped, &this->bufferData,
        sizeof(PointLights::BufferData));
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

void SpotLights::updateBuffer(const uint32_t nextImage) const
{
    memcpy(
        this->storageBuffers[nextImage].mapped, &this->bufferData,
        sizeof(SpotLights::BufferData));
}
