#include "RingBuffer.hpp"

using namespace wheels;
namespace
{

#ifndef NDEBUG
constexpr uint32_t sMaxAllocation = 0xFFFFFFFF - RingBuffer::sAlignment;
#endif // NDEBUG

} // namespace

RingBuffer::RingBuffer(
    Device *device, vk::BufferUsageFlags usage, uint32_t byteSize,
    const char *debugName)
: _device{device}
{
    assert(_device != nullptr);

    // Implementation assumes these in allocate()
    assert(byteSize > RingBuffer::sAlignment);
    assert(byteSize <= sMaxAllocation);

    _buffer = _device->createBuffer(BufferCreateInfo{
        .desc =
            BufferDescription{
                .byteSize = byteSize,
                .usage = usage,
                .properties = vk::MemoryPropertyFlagBits::eHostVisible |
                              vk::MemoryPropertyFlagBits::eHostCoherent,
            },
        .createMapped = true,
        .debugName = debugName,
    });
    assert(_buffer.mapped != nullptr);
}

RingBuffer::~RingBuffer()
{
    if (_device != nullptr)
        _device->destroy(_buffer);
}

vk::Buffer RingBuffer::buffer() const { return _buffer.handle; }

void RingBuffer::startFrame()
{
#ifndef NDEBUG
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
#endif
}

uint32_t RingBuffer::write(wheels::Span<const uint8_t> data)
{
    return write_internal(data, true);
}

void RingBuffer::write_unaligned(wheels::Span<const uint8_t> data)
{
    write_internal(data, false);
}

uint32_t RingBuffer::write_internal(
    wheels::Span<const uint8_t> data, bool align)
{
    assert(data.size() <= _buffer.byteSize);

    const uint32_t byteSize = asserted_cast<uint32_t>(data.size());
    assert(byteSize + RingBuffer::sAlignment < sMaxAllocation);

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
        assert(align && "Unaligned write wrapped around");
        _currentByteOffset = 0;
    }

    const uint32_t writeOffset = _currentByteOffset;

    uint8_t *dst = static_cast<uint8_t *>(_buffer.mapped);
    dst += writeOffset;
    memcpy(dst, data.data(), data.size());

    _currentByteOffset += byteSize;

    assert(!_frameStartOffsets.empty() && "Forgot to call startFrame()?");
    assert(
        (_frameStartOffsets.back() < _currentByteOffset ||
         _frameStartOffsets.front() > _currentByteOffset) &&
        "Stomped over an in flight frame");

    return writeOffset;
}
