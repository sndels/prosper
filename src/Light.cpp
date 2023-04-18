#include "Light.hpp"

using namespace wheels;

void DirectionalLight::bufferInfos(Span<vk::DescriptorBufferInfo> output) const
{
    assert(output.size() == uniformBuffers.size());

    size_t const size = uniformBuffers.size();
    for (size_t i = 0; i < size; ++i)
        output[i] = vk::DescriptorBufferInfo{
            .buffer = uniformBuffers[i].handle,
            .offset = 0,
            .range = sizeof(Parameters),
        };
}

void DirectionalLight::updateBuffer(const uint32_t nextImage) const
{
    memcpy(
        this->uniformBuffers[nextImage].mapped, &this->parameters,
        sizeof(Parameters));
}

void PointLights::bufferInfos(Span<vk::DescriptorBufferInfo> output) const
{
    assert(output.size() == storageBuffers.size());

    size_t const size = storageBuffers.size();
    for (size_t i = 0; i < size; ++i)
        output[i] = vk::DescriptorBufferInfo{
            .buffer = storageBuffers[i].handle,
            .offset = 0,
            .range = sBufferByteSize,
        };
}

void PointLights::updateBuffer(const uint32_t nextImage) const
{
    memcpy(
        this->storageBuffers[nextImage].mapped, this->data.data(),
        sizeof(PointLight) * this->data.size());
    const uint32_t size = asserted_cast<uint32_t>(this->data.size());
    memcpy(
        (uint8_t *)this->storageBuffers[nextImage].mapped + sBufferByteSize -
            sizeof(uint32_t),
        &size, sizeof(size));
}

void SpotLights::bufferInfos(Span<vk::DescriptorBufferInfo> output) const
{
    assert(output.size() == storageBuffers.size());

    size_t const size = storageBuffers.size();
    for (size_t i = 0; i < size; ++i)
        output[i] = vk::DescriptorBufferInfo{
            .buffer = storageBuffers[i].handle,
            .offset = 0,
            .range = sBufferByteSize,
        };
}

void SpotLights::updateBuffer(const uint32_t nextImage) const
{
    memcpy(
        this->storageBuffers[nextImage].mapped, this->data.data(),
        sizeof(SpotLight) * this->data.size());
    const uint32_t size = asserted_cast<uint32_t>(this->data.size());
    memcpy(
        (uint8_t *)this->storageBuffers[nextImage].mapped + sBufferByteSize -
            sizeof(uint32_t),
        &size, sizeof(size));
}
