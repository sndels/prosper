pushd "%~dp0.."

call scripts\cmake_compile_commands.bat

python "C:\Program Files\LLVM\bin\run-clang-tidy" -p="build-analysis" ".*\\prosper\\src\\.*.cpp"

popd
