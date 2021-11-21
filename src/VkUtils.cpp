#include "VkUtils.hpp"

void checkSuccess(vk::Result result, const char *source)
{
    if (result != vk::Result::eSuccess)
    {
        throw std::runtime_error(
            std::string(source) + " failed: " + vk::to_string(result) + "!");
    }
}
