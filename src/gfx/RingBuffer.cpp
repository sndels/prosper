#include "RingBuffer.hpp"

#include "Device.hpp"

using namespace wheels;

namespace gfx
{

namespace
{

constexpr uint32_t sMaxAllocation = 0xFFFF'FFFF - RingBuffer::sAlignment;

} // namespace

RingBuffer::~RingBuffer()
{
    // Don't check for m_initialized as we might be cleaning up after a failed
    // init.
    gDevice.destroy(m_buffer);
}

void RingBuffer::init(
    vk::BufferUsageFlags usage, uint32_t byteSize, const char *debugName)
{
    WHEELS_ASSERT(!m_initialized);

    // Implementation assumes these in allocate()
    WHEELS_ASSERT(byteSize > RingBuffer::sAlignment);
    WHEELS_ASSERT(byteSize <= sMaxAllocation);

    m_buffer = gDevice.createBuffer(BufferCreateInfo{
        .desc =
            BufferDescription{
                .byteSize = byteSize,
                .usage = usage,
                .properties = vk::MemoryPropertyFlagBits::eHostVisible |
                              vk::MemoryPropertyFlagBits::eHostCoherent,
            },
        .debugName = debugName,
    });
    WHEELS_ASSERT(m_buffer.mapped != nullptr);

    m_initialized = true;
}

vk::Buffer RingBuffer::buffer() const
{
    WHEELS_ASSERT(m_initialized);

    return m_buffer.handle;
}

void RingBuffer::startFrame()
{
    WHEELS_ASSERT(m_initialized);

    if (m_frameStartOffsets.size() < m_frameStartOffsets.capacity())
        m_frameStartOffsets.push_back(m_currentByteOffset);
    else
    {
        // This is not an efficient deque but there shouldn't be many of these
        // buffers doing this once a frame
        const size_t offsetCount = m_frameStartOffsets.size();
        for (size_t i = 0; i < offsetCount - 1; ++i)
            m_frameStartOffsets[i] = m_frameStartOffsets[i + 1];
        m_frameStartOffsets.back() = m_currentByteOffset;
    }
}

void RingBuffer::reset()
{
    WHEELS_ASSERT(m_initialized);

    m_currentByteOffset = 0;
    m_frameStartOffsets.clear();
}

uint32_t RingBuffer::write(wheels::Span<const uint8_t> data)
{
    WHEELS_ASSERT(m_initialized);

    return write_internal(data, true);
}

void RingBuffer::write_unaligned(wheels::Span<const uint8_t> data)
{
    WHEELS_ASSERT(m_initialized);

    write_internal(data, false);
}

uint32_t RingBuffer::write_internal(
    wheels::Span<const uint8_t> data, bool align)
{

    const uint32_t byteSize = asserted_cast<uint32_t>(data.size());
    WHEELS_ASSERT(byteSize + RingBuffer::sAlignment < sMaxAllocation);

    // Align offset
    if (align && (m_currentByteOffset & (RingBuffer::sAlignment - 1)) != 0)
        // Won't overflow since m_currentByteOffset is at most sMaxAllocation
        m_currentByteOffset +=
            RingBuffer::sAlignment -
            (m_currentByteOffset & (RingBuffer::sAlignment - 1));

    // Wrap around if we're out of room
    if (m_buffer.byteSize <= m_currentByteOffset ||
        m_buffer.byteSize - m_currentByteOffset < byteSize)
    {
        WHEELS_ASSERT(align && "Unaligned write wrapped around");
        m_currentByteOffset = 0;
    }

    const uint32_t writeOffset = m_currentByteOffset;

    uint8_t *dst = static_cast<uint8_t *>(m_buffer.mapped);
    dst += writeOffset;
    memcpy(dst, data.data(), data.size());

    m_currentByteOffset += byteSize;

    WHEELS_ASSERT(
        !m_frameStartOffsets.empty() && "Forgot to call startFrame()?");
    WHEELS_ASSERT(
        (m_frameStartOffsets.back() < m_currentByteOffset ||
         m_frameStartOffsets.front() > m_currentByteOffset) &&
        "Stomped over an in flight frame");

    return writeOffset;
}

} // namespace gfx
