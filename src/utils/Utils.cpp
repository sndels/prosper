#include "Utils.hpp"

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
    // Open from end to find size from initial position
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error(
            std::string{"Failed to open file '"} + path.string() + "'");

    // We won't read a file whose size won't fit size_t on a 64bit system
    const auto fileSize = static_cast<size_t>(file.tellg());
    String buffer{alloc, fileSize};
    buffer.resize(fileSize);

    // Seek to beginning and read
    file.seekg(0);
    file.read(
        reinterpret_cast<char *>(buffer.data()),
        asserted_cast<std::streamsize>(fileSize));

    file.close();
    return buffer;
}
