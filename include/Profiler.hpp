#ifndef PROSPER_PROFILER_HPP
#define PROSPER_PROFILER_HPP

#include <Device.hpp>

class GpuFrameProfiler
{
  public:
    class Scope
    {
      public:
        ~Scope();

        Scope(Scope const &) = delete;
        Scope(Scope &&other);
        Scope &operator=(Scope const &) = delete;
        Scope &operator=(Scope &&other);

      protected:
        Scope(
            vk::CommandBuffer cb, vk::QueryPool queryPool,
            const std::string &name, uint32_t index);

      private:
        vk::CommandBuffer _cb;
        vk::QueryPool _queryPool;
        uint32_t _index{0};

        friend class GpuFrameProfiler;
    };

    struct ScopeTime
    {
        uint32_t index{0xFFFFFFFF};
        float millis{0.f};
    };

    GpuFrameProfiler(Device *device);
    ~GpuFrameProfiler();

    GpuFrameProfiler(GpuFrameProfiler const &) = delete;
    GpuFrameProfiler(GpuFrameProfiler &&other);
    GpuFrameProfiler &operator=(GpuFrameProfiler const &) = delete;
    GpuFrameProfiler &operator=(GpuFrameProfiler &&other);

  protected:
    void startFrame();
    void endFrame(vk::CommandBuffer cb);

    [[nodiscard]] Scope createScope(
        vk::CommandBuffer cb, const std::string &name, uint32_t index);

    // This will read garbage if the corresponding frame index has yet to have
    // any frame complete.
    std::vector<ScopeTime> getTimes();

  private:
    Device *_device;
    Buffer _buffer;
    vk::QueryPool _queryPool;
    uint32_t _writtenQueries{0};

    friend class Profiler;
};

class Profiler
{
  public:
    class Scope
    {
      public:
        ~Scope() = default;

      private:
        Scope(GpuFrameProfiler::Scope &&gpuScope)
        : _gpuScope{std::move(gpuScope)}
        {
        }

        GpuFrameProfiler::Scope _gpuScope;

        friend class Profiler;
    };

    struct ScopeTime
    {
        std::string name;
        float gpuMillis{0.f};
    };

    Profiler(Device *device, uint32_t maxFrameCount);
    ~Profiler() = default;

    // Assume one profiler that is initialized in place
    Profiler(Profiler const &) = delete;
    Profiler(Profiler &&) = delete;
    Profiler &operator=(Profiler const &) = delete;
    Profiler &operator=(Profiler &&) = delete;

    // Should be called before any command buffer recording
    // Returns times from previous frame 'index', if any
    [[nodiscard]] std::vector<ScopeTime> startFrame(uint32_t index);
    // Should be called with the frame's presenting cb after the present barrier
    // to piggyback gpu readback sync on it.
    // Note: All scopes should end before the present barrier.
    void endFrame(vk::CommandBuffer cb);

    [[nodiscard]] Scope createScope(
        vk::CommandBuffer cb, std::string const &name);

  private:
    std::vector<Profiler::ScopeTime> getTimes(uint32_t frameIndex);

    std::vector<GpuFrameProfiler> _gpuFrameProfilers;

    uint32_t _currentFrame{0};
    std::vector<std::vector<std::string>> _scopeNames;
};

#endif // PROSPER_PROFILER_HPP
