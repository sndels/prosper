#include "ShaderIncludes.hpp"

#include "../utils/Hashes.hpp"
#include "../utils/Utils.hpp"
#include <wheels/containers/pair.hpp>

using namespace wheels;

namespace
{

const char *const sIncludePrefixCStr = "#include ";
const StrSpan sIncludePrefix{sIncludePrefixCStr};

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

    const size_t currentLength = currentSource.size();

    size_t frontCursor = 0;
    size_t backCursor = 0;
    while (frontCursor < currentLength)
    {
        // Find next potential include
        while (backCursor < currentLength && currentSource[backCursor] != '#')
            backCursor++;

        if (backCursor == currentLength)
        {
            // Reached the end so copy the remaning span
            const StrSpan frontSpan{
                &currentSource[frontCursor], backCursor - frontCursor};
            fullSource->extend(frontSpan);
            frontCursor = backCursor;
            break;
        }

        const StrSpan tailSpan{
            &currentSource[backCursor], currentLength - backCursor};

        if (!tailSpan.starts_with(sIncludePrefix))
        {
            // Continue until we have the full uninterrupted block we can memcpy
            backCursor++;
            continue;
        }

        // Let's copy what's between the cursors before the include
        const StrSpan frontSpan{
            &currentSource[frontCursor], backCursor - frontCursor};
        fullSource->extend(frontSpan);

        // Parse the include path
        const Optional<size_t> includePathFrontQuatation =
            tailSpan.find_first('"');
        if (!includePathFrontQuatation.has_value())
            throw std::runtime_error(
                std::string("Malformed shader include in '") +
                currentPath.string() + "'. Parser expects paths in '\"'s.");
        const size_t includePathStart = *includePathFrontQuatation + 1;

        const Optional<size_t> includePathNextQuotation =
            StrSpan{
                &tailSpan[includePathStart + 1],
                tailSpan.size() - includePathStart}
                .find_first('"');
        if (!includePathNextQuotation.has_value())
            throw std::runtime_error(
                std::string("Malformed shader include in '") +
                currentPath.string() + "'. Parser expects paths in '\"'s.");
        const size_t includePathLength = *includePathNextQuotation + 1;

        const StrSpan includeSpan{
            &tailSpan[includePathStart], includePathLength};
        // Need null-termination for path conversion
        const String includeRelPath{alloc, includeSpan};

        const Pair<std::filesystem::path, String> include =
            getInclude(alloc, currentPath, includeRelPath, uniqueIncludes);

        // Move cursors past the include path
        frontCursor = backCursor + includePathStart + includePathLength + 1;
        backCursor = frontCursor;

        const StrSpan fSpan{
            &currentSource[frontCursor], currentLength - backCursor};

        const std::string genericPath = include.first.generic_string();
        const StrSpan genericSpan{genericPath.data(), genericPath.size()};

        fullSource->extend("\n// Begin : ");
        fullSource->extend(genericSpan);
        fullSource->push_back('\n');
        fullSource->push_back('\n');

        expandIncludes(
            alloc, include.first.string().c_str(), include.second, fullSource,
            uniqueIncludes, includeDepth + 1);

        fullSource->extend("\n// End: ");
        fullSource->extend(genericSpan);
        fullSource->push_back('\n');
        // Second \n comes from the include newline we don't skip
    }
    WHEELS_ASSERT(frontCursor == backCursor);
}
