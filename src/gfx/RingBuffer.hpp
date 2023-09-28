#ifndef PROSPER_GFX_RING_BUFFER_HPP
#define PROSPER_GFX_RING_BUFFER_HPP

#include "../utils/Utils.hpp"
#include "Device.hpp"
#include "Resources.hpp"

#include <wheels/containers/span.hpp>
#include <wheels/containers/static_array.hpp>

class RingBuffer
{
  public:
    // Let's be safe, this is the maximum value in the wild for
    // minUniformBufferOffsetAlignment
    static const uint32_t sAlignment = 256;

    RingBuffer(Device *device, uint32_t byteSize, const char *debugName);
    ~RingBuffer();

    RingBuffer(const RingBuffer &) = delete;
    RingBuffer &operator=(const RingBuffer &) = delete;
    RingBuffer(RingBuffer &&) = delete;
    RingBuffer &operator=(RingBuffer &&) = delete;

    [[nodiscard]] vk::Buffer buffer() const;

    void startFrame();

    // The write implementations return the starting offset of the written bytes
    // in the underlying buffer.

    [[nodiscard]] uint32_t write(wheels::Span<const uint8_t> data);

    template <typename T>
        requires std::is_trivially_copyable_v<T>
    [[nodiscard]] uint32_t write_value(const T &data);

    // Writes size() elements
    template <typename T>
        requires std::is_trivially_copyable_v<T>
    [[nodiscard]] uint32_t write_elements(const wheels::Array<T> &data);

    // Writes capacity() elements
    template <typename T, size_t N>
        requires std::is_trivially_copyable_v<T>
    [[nodiscard]] uint32_t write_full_capacity(
        const wheels::StaticArray<T, N> &data);

  private:
    Device *_device{nullptr};
    Buffer _buffer;
    uint32_t _currentByteOffset{0};
#ifndef NDEBUG
    wheels::StaticArray<uint32_t, MAX_FRAMES_IN_FLIGHT - 1> _frameStartOffsets;
#endif // NDEBUG
};

template <typename T>
    requires std::is_trivially_copyable_v<T>
uint32_t RingBuffer::write_value(const T &data)
{
    return write(
        wheels::Span{reinterpret_cast<const uint8_t *>(&data), sizeof(data)});
}

template <typename T>
    requires std::is_trivially_copyable_v<T>
uint32_t RingBuffer::write_elements(const wheels::Array<T> &data)
{
    return write(wheels::Span{
        reinterpret_cast<const uint8_t *>(data.data()),
        data.size() * sizeof(T)});
}

template <typename T, size_t N>
    requires std::is_trivially_copyable_v<T>
uint32_t RingBuffer::write_full_capacity(const wheels::StaticArray<T, N> &data)
{
    return write(wheels::Span{
        reinterpret_cast<const uint8_t *>(data.data()),
        data.capacity() * sizeof(T)});
}

#endif // PROSPER_GFX_RING_BUFFER_HPP
