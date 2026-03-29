# Sindarin Tensor Libraries Installer for Windows
# Downloads and extracts the latest sindarin-pkg-tensor libs to .\libs\windows

$ErrorActionPreference = "Stop"
$REPO = "SindarinSDK/sindarin-pkg-tensor"
$BASE_DIR = Join-Path $PSScriptRoot ".." "libs"
$PLATFORM = "windows"
$INSTALL_DIR = Join-Path $BASE_DIR $PLATFORM

function Get-Architecture {
    $arch = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture
    switch ($arch) {
        "X64"   { return "x64" }
        "Arm64" { return "arm64" }
        default { throw "Unsupported architecture: $arch" }
    }
}

function Get-LatestRelease {
    param([string]$Arch)
    $apiUrl = "https://api.github.com/repos/$REPO/releases/latest"
    $headers = @{ "User-Agent" = "sindarin-installer" }
    if ($env:GITHUB_TOKEN) { $headers["Authorization"] = "Bearer $env:GITHUB_TOKEN" }
    $release = Invoke-RestMethod -Uri $apiUrl -Headers $headers
    $pattern = "windows-$Arch.zip"
    $asset = $release.assets | Where-Object { $_.name -like "*$pattern*" } | Select-Object -First 1
    if (-not $asset) { throw "No release asset found for windows-$Arch" }
    return @{ Version = $release.tag_name; Url = $asset.browser_download_url; Name = $asset.name }
}

try {
    Write-Host "Sindarin Tensor Libraries Installer" -ForegroundColor Cyan
    Write-Host "====================================" -ForegroundColor Cyan

    $arch = Get-Architecture
    Write-Host "Detected: windows ($arch)" -ForegroundColor Cyan

    $release = Get-LatestRelease -Arch $arch

    $cacheDir = Join-Path $env:USERPROFILE ".sn-cache" "downloads"
    if (-not (Test-Path $cacheDir)) { New-Item -ItemType Directory -Path $cacheDir -Force | Out-Null }
    $cached = Join-Path $cacheDir $release.Name

    if (Test-Path $cached) {
        Write-Host "Using cached $($release.Name)" -ForegroundColor Cyan
    } else {
        Write-Host "Downloading sindarin-tensor $($release.Version)..." -ForegroundColor Cyan
        Invoke-WebRequest -Uri $release.Url -OutFile $cached
    }

    if (Test-Path $INSTALL_DIR) { Remove-Item -Recurse -Force $INSTALL_DIR }
    New-Item -ItemType Directory -Path $INSTALL_DIR -Force | Out-Null

    Write-Host "Extracting to $INSTALL_DIR..." -ForegroundColor Cyan
    Expand-Archive -Path $cached -DestinationPath $INSTALL_DIR -Force

    Write-Host "Installation complete!" -ForegroundColor Green
} catch {
    Write-Host "Error: $_" -ForegroundColor Red
    exit 1
}
