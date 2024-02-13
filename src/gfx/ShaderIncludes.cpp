#include "ShaderIncludes.hpp"

#include "../utils/Hashes.hpp"
#include "../utils/Utils.hpp"
#include <wheels/containers/pair.hpp>

using namespace wheels;

namespace
{

Pair<std::filesystem::path, String> getInclude(
    Allocator &alloc, const std::filesystem::path &requesting_source,
    const String &requested_source,
    HashSet<std::filesystem::path> *uniqueIncludes)
{
    WHEELS_ASSERT(uniqueIncludes != nullptr);

    const std::filesystem::path requestingDir = requesting_source.parent_path();
    const std::filesystem::path requestedSource =
        (requestingDir / requested_source.c_str()).lexically_normal();
    WHEELS_ASSERT(std::filesystem::exists(requestedSource));

    uniqueIncludes->insert(requestedSource);

    String content = readFileString(alloc, requestedSource);

    return Pair{
        WHEELS_MOV(requestedSource),
        WHEELS_MOV(content),
    };
}

} // namespace

void expandIncludes(
    Allocator &alloc, const std::filesystem::path &currentPath,
    StrSpan currentSource, String *fullSource,
    HashSet<std::filesystem::path> *uniqueIncludes, size_t includeDepth)
{
    WHEELS_ASSERT(fullSource != nullptr);
    WHEELS_ASSERT(uniqueIncludes != nullptr);

    if (includeDepth > 100)
        throw std::runtime_error(
            std::string{"Deep shader include recursion at '"} +
            currentPath.string() + "'. Cycle?");

    const char *const includePrefix = "#include \"";
    const size_t includePrefixLength = 10;
    const size_t currentLength = currentSource.size();
    // Make sure we can avoid underflow checks on size - includePrefixLength
    WHEELS_ASSERT(currentLength > includePrefixLength);

    size_t frontCursor = 0;
    size_t backCursor = 0;
    while (frontCursor < currentLength)
    {
        // Find next potential include
        while (backCursor < currentLength && currentSource[backCursor] != '#')
            backCursor++;

        bool includeFound = false;
        if (backCursor < (currentLength - includePrefixLength))
            // Safe to do this instead of strncmp as we know we're within bounds
            includeFound = memcmp(
                               &currentSource[backCursor], includePrefix,
                               includePrefixLength) == 0;

        // Skip copy until we have the full uninterrupted block we can
        // memcpy
        if (backCursor != currentLength && !includeFound)
        {
            backCursor++;
            continue;
        }

        // We're either at the '#' of an include or one past the end of the
        // source so let's copy what's between the cursors
        const size_t copySize = backCursor - frontCursor;
        fullSource->extend(StrSpan{&currentSource[frontCursor], copySize});

        if (includeFound)
        {

            const size_t includeFrontCursor = backCursor + includePrefixLength;
            size_t includeBackCursor = includeFrontCursor;
            while (includeBackCursor < currentLength &&
                   currentSource[includeBackCursor] != '"')
                includeBackCursor++;

            // Only test for \n or the first half of \r\n for simplicity, let's
            // check for the full \r\n if it becomes an issue
            if (includeBackCursor >= currentLength - 1 ||
                currentSource[includeBackCursor] != '"' ||
                (currentSource[includeBackCursor + 1] != '\n' &&
                 (currentSource[includeBackCursor + 1] != '\r')))
                throw std::runtime_error(
                    std::string("Malformed shader include in '") +
                    currentPath.string() + "'. Parser expects paths in '\"'s.");

            // Need null-termination
            const String includeRelPath{
                alloc, StrSpan{
                           &currentSource[includeFrontCursor],
                           includeBackCursor - includeFrontCursor}};

            const Pair<std::filesystem::path, String> include =
                getInclude(alloc, currentPath, includeRelPath, uniqueIncludes);

            // Move cursors past the include line
            frontCursor = includeBackCursor + 2;
            backCursor = frontCursor;

            const std::string genericPath = include.first.generic_string();
            const StrSpan genericSpan{genericPath.data(), genericPath.size()};

            fullSource->extend("\n// Begin : ");
            fullSource->extend(genericSpan);
            fullSource->push_back('\n');
            fullSource->push_back('\n');

            expandIncludes(
                alloc, include.first.string().c_str(), include.second,
                fullSource, uniqueIncludes, includeDepth + 1);

            fullSource->extend("\n// End: ");
            fullSource->extend(genericSpan);
            fullSource->push_back('\n');
            fullSource->push_back('\n');
        }
        else
            frontCursor = backCursor;
    }
}
