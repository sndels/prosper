#include "Utils.hpp"

#include "Logger.hpp"
#include <cstring>
#include <wheels/assert.hpp>
#include <wheels/containers/static_array.hpp>

#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#else // !_WIN32

// Assume Linux
#include <sys/prctl.h>

#endif // _WIN32

using namespace wheels;

std::filesystem::path resPath(const std::filesystem::path &path)
{
    if (path.is_absolute())
        return path;
    return std::filesystem::path{RES_PATH} / path;
}

std::filesystem::path relativePath(const std::filesystem::path &path)
{
    const std::string pathStr = path.generic_string();

    // Just compare the start because this specifically doesn't consider
    // symlinks to be any different from normal folders within paths
    if (pathStr.find(RES_PATH) != std::string::npos)
        return std::filesystem::path{
            pathStr.begin() + asserted_cast<std::ptrdiff_t>(strlen(RES_PATH)),
            pathStr.end()};

    return path;
}

std::filesystem::path binPath(const std::filesystem::path &path)
{
    if (path.is_absolute())
        return path;
    return std::filesystem::path{BIN_PATH} / path;
}

String readFileString(Allocator &alloc, const std::filesystem::path &path)
{
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error(
            std::string{"Failed to open file '"} + path.string() + "'");

    // We won't read a file whose size won't fit size_t on a 64bit system
    const auto fileSize = static_cast<size_t>(file.tellg());
    String buffer{alloc, fileSize};
    buffer.resize(fileSize);

    file.seekg(0);
    file.read(
        reinterpret_cast<char *>(buffer.data()),
        asserted_cast<std::streamsize>(fileSize));

    file.close();
    return buffer;
}

void setCurrentThreadName(const char *name)
{
    // This is the prctl maximum including null
    constexpr size_t sMaxLength = 16;

    const size_t len = strlen(name);
    // Extra character for null
    WHEELS_ASSERT(len < sMaxLength && "Thread name is too long");

#ifdef _WIN32
    // This shouldn't be called a lot so let's convert the simple way
    StaticArray<wchar_t, sMaxLength> wName;
    for (size_t i = 0; i < len; ++i)
        wName[i] = name[i];
    wName[len] = '\0';

    HANDLE thread = GetCurrentThread();
    const HRESULT hr = SetThreadDescription(thread, wName.data());
    if (FAILED(hr))
        LOG_WARN("Failed to set thread name for '%s'", name);

#else // !_WIN32

    if (prctl(PR_SET_NAME, name) < 0)
        LOG_WARN("Failed to set thread name for '%s'", name);

#endif // _WIN32
}
