#include "Utils.hpp"

#include <fstream>

std::filesystem::path resPath(const std::filesystem::path &path)
{
    if (path.is_absolute())
        return path;
    return std::filesystem::path{RES_PATH} / path;
}

std::filesystem::path binPath(const std::filesystem::path &path)
{
    if (path.is_absolute())
        return path;
    return std::filesystem::path{BIN_PATH} / path;
}

std::string readFileString(const std::filesystem::path &path)
{
    // Open from end to find size from initial position
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error(
            std::string{"Failed to open file '"} + path.string() + "'");

    const auto fileSize = static_cast<size_t>(file.tellg());
    std::string buffer;
    buffer.resize(fileSize);

    // Seek to beginning and read
    file.seekg(0);
    file.read(reinterpret_cast<char *>(buffer.data()), fileSize);

    file.close();
    return buffer;
}
