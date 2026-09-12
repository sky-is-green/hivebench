# Fetch a llama.cpp release build for a specific GPU backend into
# tools\backends\<backend>\ so the studio can select it per launch:
#   POST /v1/server/start {"backend": "rocm", ...}
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File tools\fetch_backend.ps1 -Backend rocm
#
# Backends actively tested on this project's hardware: vulkan (RX 7900 XT),
# rocm (RX 7900 XT). cuda/sycl are selectable but untested here.

param(
    [Parameter(Mandatory)]
    [ValidateSet('vulkan', 'rocm', 'cuda', 'cpu')]
    [string]$Backend,
    [string]$Tag = 'latest'
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path $PSScriptRoot -Parent
$Dest = Join-Path $Root "tools\backends\$Backend"

$exe = Join-Path $Dest 'llama-server.exe'
if (Test-Path $exe) {
    Write-Host "llama-server already present for backend '$Backend' at $exe"
    exit 0
}

Write-Host "resolving llama.cpp release assets..."
$rel = if ($Tag -eq 'latest') {
    Invoke-RestMethod 'https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=15'
} else {
    @(Invoke-RestMethod "https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/$Tag")
}

# release zips are named llama-bXXXX-bin-win-<backend>-[variant-]x64.zip;
# cudart companions share the cuda suffix and must not match
$pattern = switch ($Backend) {
    'vulkan' { '^llama-.+bin-win-vulkan-x64\.zip$' }
    'rocm'   { '^llama-.+bin-win-rocm-[\d.]+-x64\.zip$' }
    'cuda'   { '^llama-.+bin-win-cuda-[\d.]+-x64\.zip$' }
    'cpu'    { '^llama-.+bin-win-cpu-x64\.zip$' }
}
$asset = $null
foreach ($release in @($rel)) {
    $asset = $release.assets |
        Where-Object { $_.name -match $pattern } | Select-Object -First 1
    if ($asset) { Write-Host "found $($asset.name) in $($release.tag_name)"; break }
}
if (-not $asset) { throw "no release asset matching $pattern in the last 15 releases" }

$zip = Join-Path $env:TEMP "llama-$Backend.zip"
New-Item -ItemType Directory -Force -Path $Dest | Out-Null
Write-Host "downloading $($asset.name) ($([Math]::Round($asset.size / 1MB)) MB)..."
Invoke-WebRequest $asset.browser_download_url -OutFile $zip
Expand-Archive $zip -DestinationPath $Dest -Force
Remove-Item $zip

# some release zips nest files one level deep; flatten so llama-server.exe
# sits directly in the backend directory
$nested = Get-ChildItem $Dest -Recurse -Filter 'llama-server.exe' | Select-Object -First 1
if ($nested -and $nested.FullName -ne $exe) {
    Get-ChildItem $nested.DirectoryName | Move-Item -Destination $Dest -Force
}

if (-not (Test-Path $exe)) { throw "llama-server.exe not found after extract" }
Write-Host "backend '$Backend' ready: $exe"
Write-Host 'start it with:  POST /v1/server/start {"backend": "' + $Backend + '", ...}'