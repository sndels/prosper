
#ifndef PROSPER_ALLOCATORS_HPP
#define PROSPER_ALLOCATORS_HPP

#include <atomic>
#include <wheels/allocators/linear_allocator.hpp>
#include <wheels/allocators/tlsf_allocator.hpp>

// These are NOT thread-safe
struct Allocators
{

    static const size_t sGeneralAllocatorSize = wheels::megabytes(512);
    static const size_t sWorldAllocatorSize = wheels::megabytes(128);

    // Enough for 4K textures, it seems. Should also be plenty for meshes as we
    // have a hard limit of 64MB for a single mesh from the default geometry
    // buffer size.
    static const size_t sLoadingScratchSize = wheels::megabytes(256);
    // Extra mem for things outside the ctx loading loop
    static const size_t sLoadingAllocatorSize =
        sLoadingScratchSize + wheels::megabytes(16);

    // NOTE:
    // References/pointers to these can already be stored to before init() is
    // called on them. Any actual access to the allocator has to happen
    // reliably after init(), of course.
    wheels::TlsfAllocator general;
    wheels::TlsfAllocator loadingWorker;
    std::atomic<size_t> loadingWorkerHighWatermark{0};
    wheels::LinearAllocator world;

    void init();
    void destroy();
};

extern Allocators gAllocators;

#endif // PROSPER_ALLOCATORS_HPP
