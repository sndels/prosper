#include "Allocators.hpp"

using namespace wheels;

// These are used everywhere and init()/destroy() order relative to other
// similar globals is handled in main()
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
Allocators gAllocators;

void Allocators::init()
{
    this->general.init(sGeneralAllocatorSize);
    this->loadingWorker.init(sLoadingAllocatorSize);
    this->world.init(sWorldAllocatorSize);
};

void Allocators::destroy()
{
    this->general.destroy();
    this->loadingWorker.destroy();
    this->world.destroy();
}
