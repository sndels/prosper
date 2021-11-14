#include "Utils.hpp"

#include <fstream>

std::string resPath(const std::string &res)
{
    return std::string{RES_PATH} + res;
}

std::string binPath(const std::string &bin)
{
    return std::string{BIN_PATH} + bin;
}

std::vector<std::byte> readFileBytes(const std::string &filename)
{
    // Open from end to find size from initial position
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error(
            std::string{"Failed to open file '"} + filename + "'");

    const auto fileSize = static_cast<size_t>(file.tellg());
    std::vector<std::byte> buffer(fileSize);

    // Seek to beginning and read
    file.seekg(0);
    file.read(reinterpret_cast<char *>(buffer.data()), fileSize);

    file.close();
    return buffer;
}

vk::ShaderModule createShaderModule(
    const vk::Device device, const std::vector<std::byte> &spv)
{
    return device.createShaderModule(vk::ShaderModuleCreateInfo{
        .codeSize = spv.size(),
        .pCode = reinterpret_cast<const uint32_t *>(spv.data())});
}
