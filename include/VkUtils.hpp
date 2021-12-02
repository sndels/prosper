#ifndef PROSPER_VKUTILS_HPP
#define PROSPER_VKUTILS_HPP

#include "vulkan.hpp"

void checkSuccess(vk::Result result, const char *source);

template <typename T, typename V> bool containsFlag(T mask, V flag)
{
    return (mask & flag) == flag;
}

template <typename T, typename V>
void assertContainsFlag(T mask, V flag, const char *errMsg)
{
    if (!containsFlag(mask, flag))
        throw std::runtime_error(errMsg);
}

#endif // PROSPER_VKUTILS_HPP
