pushd "%~dp0.."

set ARGS=-G Ninja^
 -DCMAKE_CXX_FLAGS=-fdiagnostics-color=always^
 -DPROSPER_USE_PCH=1^
 -DPROSPER_SETUP_DEPENDENCIES=1^
 -DPROSPER_ALWAYS_O2_DEPENDENCIES=1^
 -DPROSPER_MS_CRT_LEAK_CHECK=0^
 -DPROSPER_ALLOCATOR_DEBUG=0^
 -DLIVEPP_PATH=0

cmake^
 -B build_debug^
 -DCMAKE_BUILD_TYPE=Debug^
 %ARGS%

cmake^
 -B build_release^
 -DCMAKE_BUILD_TYPE=RelWithDebInfo^
 %ARGS%

popd
