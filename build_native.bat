@echo off
setlocal
REM Native MSVC (VS18) + CUDA 13.4 native Windows build driver for Infernix.
REM Based on PR #13 (docs/windows.md) with one local adaptation: dependencies come
REM from the prebuilt global vcpkg tree at C:\vcpkg (ffmpeg 9.0.1 + curl 8.21
REM + libjpeg-turbo, x64-windows) instead of manifest-mode rebuilding (ffmpeg 8.1.1 source build
REM fails here). No vcpkg toolchain file: deps are found via CMAKE_PREFIX_PATH
REM plus the local FindFFMPEG module.
REM The generator is Ninja Multi-Config (Release only), from Visual Studio's command-line tools: the
REM Visual Studio generator's layout (build-windows\apps\Release) with Ninja's scheduling, a full
REM build in 12.6 against 17.9 minutes (2026-10-28). A tree configured with another generator is
REM deleted at its next configure or build (its objects would double the tree's ~80 GB) and built anew.
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
REM vcvars64 exports its own VCPKG_ROOT (the VS-bundled tree); re-assert ours.
set "VCPKG_ROOT=C:\vcpkg"
set "VCPKG_TARGET_TRIPLET=x64-windows"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"
cd /d "%~dp0"
if "%1"=="configure" (
  call :configure
) else if "%1"=="build" (
  if not exist build-windows\build.ninja call :configure
  if errorlevel 1 exit /b 1
  cmake --build build-windows --config Release
) else (
  echo usage: build_native.bat [configure^|build]
  exit /b 2
)
exit /b %errorlevel%

:configure
if exist build-windows\CMakeCache.txt if not exist build-windows\build.ninja (
  echo build-windows was configured with another generator: deleting it for the Ninja build
  rmdir /s /q build-windows
  if exist build-windows exit /b 1
)
cmake -S . -B build-windows -G "Ninja Multi-Config" -DCMAKE_CONFIGURATION_TYPES=Release ^
  -DCMAKE_CUDA_COMPILER="%CUDA_PATH%\bin\nvcc.exe" ^
  -DVCPKG_TARGET_TRIPLET=x64-windows ^
  -DCMAKE_PREFIX_PATH="C:\vcpkg\installed\x64-windows" ^
  -DCMAKE_CUDA_ARCHITECTURES=120a ^
  -DINFERNIX_BUILD_APPS=ON ^
  -DBUILD_TESTING=ON
exit /b %errorlevel%
