#include "RingBuffer.hpp"

using namespace wheels;
namespace
{

#ifndef NDEBUG
constexpr uint32_t sMaxAllocation = 0xFFFFFFFF - RingBuffer::sAlignment;
#endif // NDEBUG

} // namespace

uint32_t RingBuffer::Allocation::byteOffset() const { return _byteOffset; }

uint32_t RingBuffer::Allocation::byteSize() const { return _byteSize; }

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
RingBuffer::Allocation::Allocation(uint32_t byteOffset, uint32_t byteSize)
: _byteOffset{byteOffset}
, _byteSize{byteSize}
{
}

RingBuffer::RingBuffer(Device *device, uint32_t byteSize, const char *debugName)
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
                .usage = vk::BufferUsageFlagBits::eStorageBuffer,
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

RingBuffer::Allocation RingBuffer::allocate(uint32_t byteSize)
{
    assert(byteSize + RingBuffer::sAlignment < sMaxAllocation);
    assert(byteSize <= _buffer.byteSize);

    // Align offset
    if ((_currentByteOffset & (RingBuffer::sAlignment - 1)) != 0)
        // Won't overflow since _currentByteOffset is at most sMaxAllocation
        _currentByteOffset +=
            RingBuffer::sAlignment -
            (_currentByteOffset & (RingBuffer::sAlignment - 1));

    // Wrap around if we're out of room
    if (_buffer.byteSize <= _currentByteOffset ||
        _buffer.byteSize - _currentByteOffset < byteSize)
        _currentByteOffset = 0;

    Allocation alloc{_currentByteOffset, byteSize};
    _currentByteOffset += byteSize;

    assert(!_frameStartOffsets.empty() && "Forgot to call startFrame()?");
    assert(
        (_frameStartOffsets.back() < _currentByteOffset ||
         _frameStartOffsets.front() > _currentByteOffset) &&
        "Stomped over an in flight frame");

    return alloc;
}

void RingBuffer::write(Allocation alloc, wheels::Span<const uint8_t> data) const
{
    assert(alloc.byteSize() == data.size());
    assert(alloc.byteOffset() < _buffer.byteSize);
    assert(_buffer.byteSize - alloc.byteOffset() >= alloc.byteSize());

    uint8_t *dst = static_cast<uint8_t *>(_buffer.mapped);
    dst += alloc.byteOffset();
    memcpy(dst, data.data(), data.size());
}
