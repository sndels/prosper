pushd "%~dp0.."

cmake^
 -B build-analysis^
 -DPROSPER_USE_PCH=OFF^
 -G Ninja^
 -S .^
 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

popd
