@echo off
setlocal
REM Incremental build of selected targets in build-windows (same environment as build_native.bat).
REM usage: build_native_target.bat <target> [<target> ...]
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
set "VCPKG_ROOT=C:\vcpkg"
set "VCPKG_TARGET_TRIPLET=x64-windows"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"
cd /d "%~dp0"
cmake --build build-windows --config Release --parallel --target %*
exit /b %errorlevel%
