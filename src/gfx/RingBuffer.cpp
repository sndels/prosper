#include "RingBuffer.hpp"

#include "Device.hpp"

using namespace wheels;
namespace
{

constexpr uint32_t sMaxAllocation = 0xFFFFFFFF - RingBuffer::sAlignment;

} // namespace

RingBuffer::~RingBuffer()
{
    // Don't check for _initialized as we might be cleaning up after a failed
    // init.
    gDevice.destroy(_buffer);
}

void RingBuffer::init(
    vk::BufferUsageFlags usage, uint32_t byteSize, const char *debugName)
{
    WHEELS_ASSERT(!_initialized);

    // Implementation assumes these in allocate()
    WHEELS_ASSERT(byteSize > RingBuffer::sAlignment);
    WHEELS_ASSERT(byteSize <= sMaxAllocation);

    _buffer = gDevice.createBuffer(BufferCreateInfo{
        .desc =
            BufferDescription{
                .byteSize = byteSize,
                .usage = usage,
                .properties = vk::MemoryPropertyFlagBits::eHostVisible |
                              vk::MemoryPropertyFlagBits::eHostCoherent,
            },
        .debugName = debugName,
    });
    WHEELS_ASSERT(_buffer.mapped != nullptr);

    _initialized = true;
}

vk::Buffer RingBuffer::buffer() const
{
    WHEELS_ASSERT(_initialized);

    return _buffer.handle;
}

void RingBuffer::startFrame()
{
    WHEELS_ASSERT(_initialized);

    if (_frameStartOffsets.size() < _frameStartOffsets.capacity())
        _frameStartOffsets.push_back(_currentByteOffset);
    else
    {
        // This is not an efficient deque but there shouldn't be many of these
        // buffers doing this once a frame
        const size_t offsetCount = _frameStartOffsets.size();
        for (size_t i = 0; i < offsetCount - 1; ++i)
            _frameStartOffsets[i] = _frameStartOffsets[i + 1];
        _frameStartOffsets.back() = _currentByteOffset;
    }
}

void RingBuffer::reset()
{
    WHEELS_ASSERT(_initialized);

    _currentByteOffset = 0;
    _frameStartOffsets.clear();
}

uint32_t RingBuffer::write(wheels::Span<const uint8_t> data)
{
    WHEELS_ASSERT(_initialized);

    return write_internal(data, true);
}

void RingBuffer::write_unaligned(wheels::Span<const uint8_t> data)
{
    WHEELS_ASSERT(_initialized);

    write_internal(data, false);
}

uint32_t RingBuffer::write_internal(
    wheels::Span<const uint8_t> data, bool align)
{

    const uint32_t byteSize = asserted_cast<uint32_t>(data.size());
    WHEELS_ASSERT(byteSize + RingBuffer::sAlignment < sMaxAllocation);

    // Align offset
    if (align && (_currentByteOffset & (RingBuffer::sAlignment - 1)) != 0)
        // Won't overflow since _currentByteOffset is at most sMaxAllocation
        _currentByteOffset +=
            RingBuffer::sAlignment -
            (_currentByteOffset & (RingBuffer::sAlignment - 1));

    // Wrap around if we're out of room
    if (_buffer.byteSize <= _currentByteOffset ||
        _buffer.byteSize - _currentByteOffset < byteSize)
    {
        WHEELS_ASSERT(align && "Unaligned write wrapped around");
        _currentByteOffset = 0;
    }

    const uint32_t writeOffset = _currentByteOffset;

    uint8_t *dst = static_cast<uint8_t *>(_buffer.mapped);
    dst += writeOffset;
    memcpy(dst, data.data(), data.size());

    _currentByteOffset += byteSize;

    WHEELS_ASSERT(
        !_frameStartOffsets.empty() && "Forgot to call startFrame()?");
    WHEELS_ASSERT(
        (_frameStartOffsets.back() < _currentByteOffset ||
         _frameStartOffsets.front() > _currentByteOffset) &&
        "Stomped over an in flight frame");

    return writeOffset;
}
