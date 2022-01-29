#include "Utils.hpp"

#include <fstream>

std::filesystem::path resPath(const std::filesystem::path &path)
{
    if (path.is_absolute())
        return path;
    else
        return std::filesystem::path{RES_PATH} / path;
}

std::filesystem::path binPath(const std::filesystem::path &path)
{
    if (path.is_absolute())
        return path;
    else
        return std::filesystem::path{BIN_PATH} / path;
}

std::vector<std::byte> readFileBytes(const std::filesystem::path &path)
{
    // Open from end to find size from initial position
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error(
            std::string{"Failed to open file '"} + path.string() + "'");

    const auto fileSize = static_cast<size_t>(file.tellg());
    std::vector<std::byte> buffer(fileSize);

    // Seek to beginning and read
    file.seekg(0);
    file.read(reinterpret_cast<char *>(buffer.data()), fileSize);

    file.close();
    return buffer;
}

vk::ShaderModule createShaderModule(
    const vk::Device device, const std::string &debugName,
    const std::vector<std::byte> &spv)
{
    const auto sm = device.createShaderModule(vk::ShaderModuleCreateInfo{
        .codeSize = spv.size(),
        .pCode = reinterpret_cast<const uint32_t *>(spv.data())});

    device.setDebugUtilsObjectNameEXT(vk::DebugUtilsObjectNameInfoEXT{
        .objectType = vk::ObjectType::eShaderModule,
        .objectHandle =
            reinterpret_cast<uint64_t>(static_cast<VkShaderModule>(sm)),
        .pObjectName = debugName.c_str(),
    });

    return sm;
}
