#ifndef PROSPER_GFX_RING_BUFFER_HPP
#define PROSPER_GFX_RING_BUFFER_HPP

#include "gfx/Fwd.hpp"
#include "gfx/Resources.hpp"
#include "utils/Utils.hpp"

#include <wheels/containers/inline_array.hpp>
#include <wheels/containers/span.hpp>

namespace gfx
{

class RingBuffer
{
  public:
    // Let's be safe, this is the maximum value in the wild for
    // minUniformBufferOffsetAlignment
    static const uint32_t sAlignment = 256;

    RingBuffer() noexcept = default;
    ~RingBuffer();

    RingBuffer(const RingBuffer &) = delete;
    RingBuffer &operator=(const RingBuffer &) = delete;
    RingBuffer(RingBuffer &&) = delete;
    RingBuffer &operator=(RingBuffer &&) = delete;

    void init(
        vk::BufferUsageFlags usage, uint32_t byteSize, const char *debugName);

    [[nodiscard]] vk::Buffer buffer() const;

    void startFrame();
    // Zeroes the write offset and debug tracking
    void reset();

    // The write implementations return the starting offset of the written bytes
    // in the underlying buffer. Unaligned writes are intended to be used to
    // append tightly after an aligned write.

    [[nodiscard]] uint32_t write(wheels::Span<const uint8_t> data);
    void write_unaligned(wheels::Span<const uint8_t> data);

    template <typename T>
        requires std::is_trivially_copyable_v<T>
    [[nodiscard]] uint32_t write_value(const T &data);
    template <typename T>
        requires std::is_trivially_copyable_v<T>
    void write_value_unaligned(const T &data);

    // Writes size() elements
    template <typename T>
        requires std::is_trivially_copyable_v<T>
    [[nodiscard]] uint32_t write_elements(const wheels::Array<T> &data);
    template <typename T>
        requires std::is_trivially_copyable_v<T>
    void write_elements_unaligned(const wheels::Array<T> &data);

    // Writes capacity() elements
    template <typename T, size_t N>
        requires std::is_trivially_copyable_v<T>
    [[nodiscard]] uint32_t write_full_capacity(
        const wheels::InlineArray<T, N> &data);
    template <typename T, size_t N>
        requires std::is_trivially_copyable_v<T>
    void write_full_capacity_unaligned(const wheels::InlineArray<T, N> &data);

  private:
    uint32_t write_internal(wheels::Span<const uint8_t> data, bool aligned);

    bool m_initialized{false};
    Buffer m_buffer;
    uint32_t m_currentByteOffset{0};
    wheels::InlineArray<uint32_t, MAX_FRAMES_IN_FLIGHT - 1> m_frameStartOffsets;
};

template <typename T>
    requires std::is_trivially_copyable_v<T>
uint32_t RingBuffer::write_value(const T &data)
{
    return write_internal(
        wheels::Span{reinterpret_cast<const uint8_t *>(&data), sizeof(data)},
        true);
}

template <typename T>
    requires std::is_trivially_copyable_v<T>
void RingBuffer::write_value_unaligned(const T &data)
{
    WHEELS_ASSERT(m_initialized);

    write_internal(
        wheels::Span{reinterpret_cast<const uint8_t *>(&data), sizeof(data)},
        false);
}

template <typename T>
    requires std::is_trivially_copyable_v<T>
uint32_t RingBuffer::write_elements(const wheels::Array<T> &data)
{
    WHEELS_ASSERT(m_initialized);

    return write_internal(
        wheels::Span{
            reinterpret_cast<const uint8_t *>(data.data()),
            data.size() * sizeof(T)},
        true);
}

template <typename T>
    requires std::is_trivially_copyable_v<T>
void RingBuffer::write_elements_unaligned(const wheels::Array<T> &data)
{
    WHEELS_ASSERT(m_initialized);

    write_internal(
        wheels::Span{
            reinterpret_cast<const uint8_t *>(data.data()),
            data.size() * sizeof(T)},
        false);
}

template <typename T, size_t N>
    requires std::is_trivially_copyable_v<T>
uint32_t RingBuffer::write_full_capacity(const wheels::InlineArray<T, N> &data)
{
    WHEELS_ASSERT(m_initialized);

    return write_internal(
        wheels::Span{
            reinterpret_cast<const uint8_t *>(data.data()),
            data.capacity() * sizeof(T)},
        true);
}

template <typename T, size_t N>
    requires std::is_trivially_copyable_v<T>
void RingBuffer::write_full_capacity_unaligned(
    const wheels::InlineArray<T, N> &data)
{
    WHEELS_ASSERT(m_initialized);

    write_internal(
        wheels::Span{
            reinterpret_cast<const uint8_t *>(data.data()),
            data.capacity() * sizeof(T)},
        false);
}

} // namespace gfx

#endif // PROSPER_GFX_RING_BUFFER_HPP
