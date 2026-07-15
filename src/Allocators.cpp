#include "Allocators.hpp"

#include "utils/TaskScheduler.hpp"

using namespace wheels;

namespace
{
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
thread_local wheels::LinearAllocator sThreadAllocator;
} // namespace

// These are used everywhere and init()/destroy() order relative to other
// similar globals is handled in main()
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
Allocators gAllocators;

// Thread allocator lifetime is tied to Allocators::init()/destroy()
// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
wheels::LinearAllocator &Allocators::threadAllocator() const
{
    return sThreadAllocator;
}

void Allocators::init()
{
    this->general.init(sGeneralAllocatorSize);
    this->world.init(sWorldAllocatorSize);

    enki::LambdaPinnedTask initThreadAllocator(
        []() { sThreadAllocator.init(Allocators::sThreadAllocatorSize); });
    const uint32_t threadCount = utils::gTaskScheduler.GetNumTaskThreads();
    for (uint32_t i = 0; i < threadCount; ++i)
    {
        initThreadAllocator.threadNum = i;
        utils::gTaskScheduler.AddPinnedTask(&initThreadAllocator);
        utils::gTaskScheduler.WaitforTask(&initThreadAllocator);
    }
};

void Allocators::destroy()
{
    this->general.destroy();
    this->world.destroy();

    enki::LambdaPinnedTask destroyThreadAllocator(
        []() { sThreadAllocator.destroy(); });
    WHEELS_ASSERT(
        utils::gTaskScheduler.GetNumTaskThreads() ==
        enki::GetNumHardwareThreads());
    const uint32_t threadCount = enki::GetNumHardwareThreads();
    for (uint32_t i = 0; i < threadCount; ++i)
    {
        destroyThreadAllocator.threadNum = i;
        utils::gTaskScheduler.AddPinnedTask(&destroyThreadAllocator);
        utils::gTaskScheduler.WaitforTask(&destroyThreadAllocator);
    }
}
