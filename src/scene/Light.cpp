#include "Light.hpp"

#include "gfx/RingBuffer.hpp"

using namespace wheels;

namespace scene
{

uint32_t DirectionalLight::write(gfx::RingBuffer &buffer) const
{
    return buffer.write_value(this->parameters);
}

uint32_t PointLights::write(gfx::RingBuffer &buffer) const
{
    const uint32_t offset = buffer.write_full_capacity(this->data);

    const uint32_t size = asserted_cast<uint32_t>(this->data.size());
    buffer.write_value_unaligned(size);

    return offset;
}

uint32_t SpotLights::write(gfx::RingBuffer &buffer) const
{
    const uint32_t offset = buffer.write_full_capacity(this->data);

    const uint32_t size = asserted_cast<uint32_t>(this->data.size());
    buffer.write_value_unaligned(size);

    return offset;
}

} // namespace scene
