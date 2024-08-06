if not exist "%~dp0..\build" mkdir %~dp0..\build
pushd "%~dp0..\build"

msbuild^
 /property:GenerateFullPaths=true^
 /p:BuildProjectReferences=false^
 /t:build^
 /consoleloggerparameters:NoSummary^
 /property:Configuration=%1^
 prosper.vcxproj

popd
