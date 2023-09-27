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

    class Allocation
    {
      public:
        Allocation() = default;
        ~Allocation() = default;

        Allocation(const Allocation &) = default;
        Allocation(Allocation &&) = default;
        Allocation &operator=(const Allocation &) = default;
        Allocation &operator=(Allocation &&) = default;

        [[nodiscard]] uint32_t byteOffset() const;
        [[nodiscard]] uint32_t byteSize() const;

        friend class RingBuffer;

      protected:
        // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
        Allocation(uint32_t byteOffset, uint32_t byteSize);

      private:
        uint32_t _byteOffset{0};
        uint32_t _byteSize{0};
    };

    RingBuffer(Device *device, uint32_t byteSize, const char *debugName);
    ~RingBuffer();

    RingBuffer(const RingBuffer &) = delete;
    RingBuffer &operator=(const RingBuffer &) = delete;
    RingBuffer(RingBuffer &&) = delete;
    RingBuffer &operator=(RingBuffer &&) = delete;

    [[nodiscard]] vk::Buffer buffer() const;

    void startFrame();

    [[nodiscard]] Allocation allocate(uint32_t byteSize);
    template <typename T> void write(Allocation alloc, const T &data) const;
    void write(Allocation alloc, wheels::Span<const uint8_t> data) const;

  private:
    Device *_device{nullptr};
    Buffer _buffer;
    uint32_t _currentByteOffset{0};
#ifndef NDEBUG
    wheels::StaticArray<uint32_t, MAX_FRAMES_IN_FLIGHT - 1> _frameStartOffsets;
#endif // NDEBUG
};

template <typename T>
void RingBuffer::write(Allocation alloc, const T &data) const
{
    write(
        alloc,
        wheels::Span{reinterpret_cast<const uint8_t *>(&data), sizeof(data)});
}

#endif // PROSPER_GFX_RING_BUFFER_HPP
