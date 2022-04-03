// MSVC /external -stuff doesn't apply here so drop warnings manually
#ifdef _MSC_VER
#pragma warning(push, 0)
#endif // _MSC_VER

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#ifdef _MSC_VER
#pragma warning(pop)
#endif // _MSC_VER
