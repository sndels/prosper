#include "ShaderIncludes.hpp"

#include "../utils/Hashes.hpp"
#include "../utils/Utils.hpp"
#include <charconv>
#include <wheels/containers/pair.hpp>
#include <wheels/containers/static_array.hpp>

using namespace wheels;

// This not the most robust parser but it handles many common error cases and
// shaderc catches some more

namespace
{

const char *const sIncludePrefixCStr = "#include ";
const StrSpan sIncludePrefix{sIncludePrefixCStr};

const char *const sLinePrefixCStr = "#line ";
const StrSpan sLinePrefix{sLinePrefixCStr};

Pair<std::filesystem::path, String> getInclude(
    Allocator &alloc, const std::filesystem::path &requesting_source,
    const String &requested_source,
    HashSet<std::filesystem::path> *uniqueIncludes)
{
    WHEELS_ASSERT(uniqueIncludes != nullptr);

    const std::filesystem::path requestingDir = requesting_source.parent_path();
    const std::filesystem::path requestedSource =
        (requestingDir / requested_source.c_str()).lexically_normal();
    if (!std::filesystem::exists(requestedSource))
        throw std::runtime_error(
            std::string{"Could not find '"} + requestedSource.generic_string() +
            '\'');

    uniqueIncludes->insert(requestedSource);

    String content = readFileString(alloc, requestedSource);

    return Pair{
        WHEELS_MOV(requestedSource),
        WHEELS_MOV(content),
    };
}

bool isAtNewline(StrSpan span)
{
    if (span.empty())
        return false;

    if (span[0] == '\n')
        return true;

    if (span[0] == '\r')
    {
        if (span.size() > 1 && span[1] == '\n')
            return true;
        // Let's not handle the pre-osx case of \r only
    }

    return false;
}

// Returns the offset over the newline at the beginning of 'span'
size_t skipNewline(StrSpan span)
{
    WHEELS_ASSERT(isAtNewline(span));
    if (span[0] == '\n')
        return 1;

    // Let's not handle the pre-osx case of \r only
    WHEELS_ASSERT(span[0] == '\r' && span[1] == '\n');
    return 2;
}

uint32_t parseLineNumber(StrSpan span)
{
    WHEELS_ASSERT(span.starts_with(sLinePrefix));

    size_t front = sLinePrefix.size();
    while (std::isspace(span[front]) != 0 && front < span.size())
    {
        if (isAtNewline(StrSpan{span.data() + front, span.size() - front}))
            throw std::runtime_error("Unexpected newline");

        front++;
    }
    WHEELS_ASSERT(front != span.size());

    uint32_t ret = 0;
    const std::from_chars_result result =
        std::from_chars(span.data() + front, span.data() + span.size(), ret);
    if (result.ec != std::errc())
        throw std::runtime_error("Failed to parse line number");

    if (ret == 0)
        throw std::runtime_error("Line number should be greater than 0");

    // Check that the second potential componnt is not there
    StrSpan tailSpan{span.data() + front, span.size() - front};
    // First go through the number part
    while (!tailSpan.empty() && std::isspace(tailSpan[0]) == 0)
        tailSpan = StrSpan{tailSpan.data() + 1, tailSpan.size() - 1};
    // Then try to find the newline
    while (!tailSpan.empty() && std::isspace(tailSpan[0]) != 0 &&
           !isAtNewline(tailSpan))
        tailSpan = StrSpan{tailSpan.data() + 1, tailSpan.size() - 1};
    if (!isAtNewline(tailSpan))
        throw std::runtime_error("Line directives support line number only");

    return ret;
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
            currentPath.generic_string() +
            " Deep shader include recursion, cycle?");

    const size_t currentLength = currentSource.size();

    const std::string genericCurrenPath = currentPath.generic_string();
    const StrSpan genericCurrentSpan{
        genericCurrenPath.data(), genericCurrenPath.size()};

    size_t frontCursor = 0;
    size_t backCursor = 0;
    uint32_t lineNumber = 1;
    bool hashFoundOnLine = false;
    while (frontCursor < currentLength)
    {
        // Find next potential include
        while (backCursor < currentLength)
        {
            if (currentSource[backCursor] == '#')
            {
                if (hashFoundOnLine)
                    throw std::runtime_error(
                        currentPath.generic_string() + ':' +
                        std::to_string(lineNumber) +
                        " Two #'s found on one line. Invalid preprocessor "
                        "directives?");
                hashFoundOnLine = true;
                break;
            }

            const StrSpan tailSpan{
                &currentSource[backCursor], currentLength - backCursor};
            if (isAtNewline(tailSpan))
            {
                hashFoundOnLine = false;
                lineNumber++;
                backCursor += skipNewline(tailSpan);
            }
            else
                backCursor++;
        }

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
            // Keep line count on track
            if (tailSpan.starts_with(sLinePrefix))
            {
                try
                {
                    // Subtract one because the following newline will increment
                    lineNumber = parseLineNumber(tailSpan) - 1;
                }
                catch (const std::exception &e)
                {
                    throw std::runtime_error(
                        currentPath.generic_string() + ':' +
                        std::to_string(lineNumber) + ' ' + e.what());
                }
            }

            // Continue until we have the full uninterrupted block we can
            // memcpy
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
                currentPath.generic_string() + ':' +
                std::to_string(lineNumber) + " Parser expects relative paths.");
        const size_t includePathStart = *includePathFrontQuatation + 1;

        const Optional<size_t> includePathNextQuotation =
            StrSpan{
                &tailSpan[includePathStart + 1],
                tailSpan.size() - includePathStart}
                .find_first('"');
        if (!includePathNextQuotation.has_value())
            throw std::runtime_error(
                currentPath.generic_string() + ':' +
                std::to_string(lineNumber) + " Parser expects relative paths.");
        const size_t includePathLength = *includePathNextQuotation + 1;

        const StrSpan includeSpan{
            &tailSpan[includePathStart], includePathLength};
        // Need null-termination for path conversion
        const String includeRelPath{alloc, includeSpan};

        Optional<Pair<std::filesystem::path, String>> include;
        try
        {
            include =
                getInclude(alloc, currentPath, includeRelPath, uniqueIncludes);
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error(
                currentPath.generic_string() + ':' +
                std::to_string(lineNumber) + ' ' + e.what());
        }
        WHEELS_ASSERT(include.has_value());

        const std::string genericIncludePath = include->first.generic_string();
        const StrSpan genericIncludeSpan{
            genericIncludePath.data(), genericIncludePath.size()};

        // Tag include source for error reporting
        fullSource->extend("\n#line 1 \"");
        fullSource->extend(genericIncludeSpan);
        fullSource->push_back('"');
        fullSource->push_back('\n');

        expandIncludes(
            alloc, include->first.string().c_str(), include->second, fullSource,
            uniqueIncludes, includeDepth + 1);

        WHEELS_ASSERT(lineNumber < 999999);
        StaticArray<char, 7> lineNumberStr;
        snprintf(lineNumberStr.data(), 7, "%u", lineNumber + 1);

        // Tag current source for error reporting
        fullSource->extend("\n#line ");
        fullSource->extend(lineNumberStr.data());
        fullSource->push_back(' ');
        fullSource->push_back('"');
        fullSource->extend(genericCurrentSpan);
        fullSource->push_back('"');
        // No newline as we don't skip the one after the include directive

        // Move cursors past the include path
        frontCursor = backCursor + includePathStart + includePathLength + 1;
        backCursor = frontCursor;
    }
    WHEELS_ASSERT(frontCursor == backCursor);
}
