# Sindarin native library installer for Windows
# Downloads the latest release from GitHub to ./libs/windows
# Caches archives in ~/.sn-cache/downloads/ to avoid re-downloading

$ErrorActionPreference = "Stop"

$REPO = "SindarinSDK/sindarin-pkg-tensor"
$PKG_NAME = "sindarin-tensor"
$INSTALL_DIR = Join-Path (Get-Location) "libs\windows"

function Write-Status {
    param(
        [string]$Message,
        [string]$Type = "Info"
    )

    $color = switch ($Type) {
        "Info" { "Cyan" }
        "Success" { "Green" }
        "Warning" { "Yellow" }
        "Error" { "Red" }
        default { "White" }
    }

    Write-Host $Message -ForegroundColor $color
}

function Get-Architecture {
    $arch = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture
    switch ($arch) {
        "X64"   { return "x64" }
        "Arm64" { return "arm64" }
        default { throw "Unsupported architecture: $arch" }
    }
}

function Get-LatestWindowsRelease {
    param([string]$Arch)

    Write-Status "Fetching latest release information..."

    $apiUrl = "https://api.github.com/repos/$REPO/releases/latest"
    $headers = @{ "User-Agent" = "sindarin-installer" }
    if ($env:GITHUB_TOKEN) { $headers["Authorization"] = "Bearer $env:GITHUB_TOKEN" }

    try {
        $release = Invoke-RestMethod -Uri $apiUrl -Headers $headers

        $pattern = "windows-$Arch.zip"
        $asset = $release.assets | Where-Object { $_.name -like "*$pattern*" } | Select-Object -First 1

        if (-not $asset) {
            throw "No Windows release asset found for windows-$Arch"
        }

        return @{
            Url = $asset.browser_download_url
            Name = $asset.name
            Version = $release.tag_name
        }
    }
    catch {
        Write-Status "Failed to fetch release info: $_" -Type "Error"
        exit 1
    }
}

function Install-SindarinLibs {
    param(
        [hashtable]$Release
    )

    # Check package cache first
    $cacheDir = Join-Path $env:USERPROFILE ".sn-cache" "downloads"
    $cachedZip = Join-Path $cacheDir $Release.Name

    if (Test-Path $cachedZip) {
        Write-Status "Using cached $($Release.Name)"
    }
    else {
        Write-Status "Downloading $PKG_NAME $($Release.Version)..."

        if (-not (Test-Path $cacheDir)) {
            New-Item -ItemType Directory -Path $cacheDir -Force | Out-Null
        }

        try {
            $dlHeaders = @{ "User-Agent" = "sindarin-installer" }
            if ($env:GITHUB_TOKEN) { $dlHeaders["Authorization"] = "Bearer $env:GITHUB_TOKEN" }
            Invoke-WebRequest -Uri $Release.Url -OutFile $cachedZip -Headers $dlHeaders -UseBasicParsing
        }
        catch {
            Write-Status "Download failed: $_" -Type "Error"
            if (Test-Path $cachedZip) {
                Remove-Item -Force $cachedZip -ErrorAction SilentlyContinue
            }
            exit 1
        }
    }

    $tempDir = Join-Path $env:TEMP "$PKG_NAME-install"

    if (Test-Path $tempDir) {
        Remove-Item -Recurse -Force $tempDir
    }
    New-Item -ItemType Directory -Path $tempDir | Out-Null

    try {
        Write-Status "Extracting to $INSTALL_DIR..."

        if (Test-Path $INSTALL_DIR) {
            Remove-Item -Recurse -Force $INSTALL_DIR
        }
        New-Item -ItemType Directory -Path $INSTALL_DIR | Out-Null

        $extractDir = Join-Path $tempDir "extracted"
        Expand-Archive -Path $cachedZip -DestinationPath $extractDir -Force

        # Handle potentially nested directory structure
        $contents = Get-ChildItem -Path $extractDir
        if ($contents.Count -eq 1 -and $contents[0].PSIsContainer) {
            $innerDir = $contents[0].FullName
            Get-ChildItem -Path $innerDir | Move-Item -Destination $INSTALL_DIR
        }
        else {
            Get-ChildItem -Path $extractDir | Move-Item -Destination $INSTALL_DIR
        }

        Write-Status "Successfully installed $PKG_NAME $($Release.Version) to $INSTALL_DIR" -Type "Success"
    }
    catch {
        Write-Status "Installation failed: $_" -Type "Error"
        exit 1
    }
    finally {
        if (Test-Path $tempDir) {
            Remove-Item -Recurse -Force $tempDir -ErrorAction SilentlyContinue
        }
    }
}

# Main execution
Write-Status "$PKG_NAME - native library installer" -Type "Info"
Write-Status "========================================" -Type "Info"

$arch = Get-Architecture
Write-Status "Detected: windows ($arch)" -Type "Info"

$release = Get-LatestWindowsRelease -Arch $arch
Install-SindarinLibs -Release $release

Write-Status ""
Write-Status "Installation complete!" -Type "Success"
