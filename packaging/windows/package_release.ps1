# Packages a Windows release of Infernix: the chosen executables, every DLL beside them, the Infernix
# licence, THIRD_PARTY_NOTICES.md and the licence of every component they contain or load, plus the
# corresponding source of the FFmpeg DLLs (LGPL) as a second zip. A DLL the notices do not cover
# stops the packaging.
#
#   powershell -ExecutionPolicy Bypass -File packaging\windows\package_release.ps1 [-Executables infernix-serve.exe,infernix.exe]
#
# Output in -OutDir (default dist): infernix-<version>-windows-x64.zip and ffmpeg-<version>-source.zip.
param(
    [string]$BuildDir    = '',
    [string]$OutDir      = '',
    [string]$VcpkgRoot   = '',
    [string]$Triplet     = 'x64-windows',
    [string[]]$Executables = @('infernix-serve.exe')
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path "$PSScriptRoot\..\..").Path
if (-not $BuildDir) { $BuildDir = "$repo\build-windows\apps\Release" }
if (-not $OutDir) { $OutDir = "$repo\dist" }
if (-not $VcpkgRoot) { $VcpkgRoot = if ($env:VCPKG_ROOT) { $env:VCPKG_ROOT } else { 'C:\vcpkg' } }
$installed = "$VcpkgRoot\installed\$Triplet"

# Every shipped DLL must belong to one of these vcpkg packages (their copyright files are the licences).
$dllPackages = [ordered]@{
    '^(avcodec|avformat|avutil|swresample|swscale)-\d+\.dll$' = 'ffmpeg'
    '^libcurl\.dll$'                                          = 'curl'
    '^z\.dll$'                                                = 'zlib'
}
$packageLicence = @{ ffmpeg = 'FFmpeg-LICENSE.txt'; curl = 'curl-LICENSE.txt'; zlib = 'zlib-LICENSE.txt' }

# Licences of the code compiled into the executables (see THIRD_PARTY_NOTICES.md).
$compiled = [ordered]@{
    'LICENSE'                                                     = 'Infernix-LICENSE.txt'
    'third_party\xgrammar\LICENSE'                                = 'xgrammar-LICENSE.txt'
    'third_party\xgrammar\NOTICE'                                 = 'xgrammar-NOTICE.txt'
    'third_party\xgrammar\3rdparty\dlpack\LICENSE'                = 'dlpack-LICENSE.txt'
    'third_party\xgrammar\3rdparty\picojson\LICENSE'              = 'picojson-LICENSE.txt'
    'third_party\llama-jinja\LICENSE'                             = 'llama-jinja-LICENSE.txt'
    'third_party\llama-jinja\UNICODE-LICENSE'                     = 'llama-jinja-UNICODE-LICENSE.txt'
    'third_party\cpp-httplib\LICENSE'                             = 'cpp-httplib-LICENSE.txt'
    'third_party\nlohmann\LICENSE.MIT'                            = 'nlohmann-json-LICENSE.txt'
    'third_party\spdlog\LICENSE'                                  = 'spdlog-LICENSE.txt'
    'third_party\spdlog\include\spdlog\fmt\bundled\fmt.license.rst' = 'fmt-LICENSE.txt'
    'third_party\utf8proc\LICENSE.md'                             = 'utf8proc-LICENSE.txt'
}

foreach ($exe in $Executables) {
    if (-not (Test-Path "$BuildDir\$exe")) { throw "$BuildDir\$exe not found: build the Release apps first" }
}
# The executable prints its stamped build id on startup; an unknown option then stops it at once.
# Through cmd: Windows PowerShell turns a native program's stderr into errors under 'Stop'.
$banner = (& cmd.exe /c "`"$BuildDir\$($Executables[0])`" --print-build-id 2>&1" | Out-String)
if ($banner -notmatch 'build (\S+)') { throw "no build id in the output of $($Executables[0])" }
$build = $Matches[1]
if ($build -like '*-dirty') { throw "build $build has uncommitted changes: commit and rebuild before packaging" }

$name  = "infernix-$($build -replace '^windows-', '')-windows-x64"
$stage = "$OutDir\$name"
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Force "$stage\licenses" | Out-Null

foreach ($exe in $Executables) { Copy-Item "$BuildDir\$exe" $stage }
$needed = @{}
foreach ($dll in Get-ChildItem "$BuildDir\*.dll") {
    $package = $null
    foreach ($pattern in $dllPackages.Keys) { if ($dll.Name -match $pattern) { $package = $dllPackages[$pattern] } }
    if (-not $package) { throw "$($dll.Name) is not covered by THIRD_PARTY_NOTICES.md: add it before shipping" }
    Copy-Item $dll.FullName $stage
    $needed[$package] = $true
}
foreach ($package in $needed.Keys) {
    $copyright = "$installed\share\$package\copyright"
    if (-not (Test-Path $copyright)) { throw "no licence for $package at $copyright" }
    Copy-Item $copyright "$stage\licenses\$($packageLicence[$package])"
}
foreach ($source in $compiled.Keys) { Copy-Item "$repo\$source" "$stage\licenses\$($compiled[$source])" }
Copy-Item "$repo\LICENSE" "$stage\LICENSE.txt"
Copy-Item "$PSScriptRoot\THIRD_PARTY_NOTICES.md" $stage

$zip = "$OutDir\$name.zip"
if (Test-Path $zip) { Remove-Item -Force $zip }
Compress-Archive -Path $stage -DestinationPath $zip

# FFmpeg's corresponding source: the source release vcpkg built and the port that built it.
if ($needed['ffmpeg']) {
    $list = Get-ChildItem "$VcpkgRoot\installed\vcpkg\info\ffmpeg_*_$Triplet.list" | Select-Object -First 1
    if (-not $list -or $list.Name -notmatch '^ffmpeg_([^_]+)_') { throw 'cannot find the installed FFmpeg version' }
    $version = $Matches[1]
    $tarball = Get-ChildItem "$VcpkgRoot\downloads\ffmpeg-ffmpeg-n$version.tar.*" | Select-Object -First 1
    if (-not $tarball) { throw "FFmpeg $version source not in $VcpkgRoot\downloads: run vcpkg install ffmpeg again" }
    $portVersion = (Get-Content -Raw "$VcpkgRoot\ports\ffmpeg\vcpkg.json" | ConvertFrom-Json).version
    if ($portVersion -ne $version) { throw "the vcpkg ffmpeg port is $portVersion but $version is installed" }
    $sourceStage = "$OutDir\ffmpeg-$version-source"
    if (Test-Path $sourceStage) { Remove-Item -Recurse -Force $sourceStage }
    New-Item -ItemType Directory -Force "$sourceStage\vcpkg-port" | Out-Null
    Copy-Item $tarball.FullName $sourceStage
    Copy-Item "$VcpkgRoot\ports\ffmpeg\*" "$sourceStage\vcpkg-port"
    Copy-Item "$installed\share\ffmpeg\copyright" "$sourceStage\LICENSE.txt"
    $sourceZip = "$OutDir\ffmpeg-$version-source.zip"
    if (Test-Path $sourceZip) { Remove-Item -Force $sourceZip }
    Compress-Archive -Path $sourceStage -DestinationPath $sourceZip
    Remove-Item -Recurse -Force $sourceStage
    "FFmpeg source: $sourceZip"
}
Remove-Item -Recurse -Force $stage
"Release: $zip"
