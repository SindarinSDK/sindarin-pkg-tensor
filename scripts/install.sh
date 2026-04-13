#!/bin/bash
# Sindarin native library installer for Linux/macOS
# Downloads the latest release from GitHub to ./libs/{os}
# Caches archives in ~/.sn-cache/downloads/ to avoid re-downloading

set -e

REPO="SindarinSDK/sindarin-pkg-tensor"
PKG_NAME="sindarin-tensor"
BASE_DIR="$(pwd)/libs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

write_status() {
    local message="$1"
    local type="${2:-info}"

    case "$type" in
        "info")
            echo -e "${CYAN}${message}${NC}"
            ;;
        "success")
            echo -e "${GREEN}${message}${NC}"
            ;;
        "warning")
            echo -e "${YELLOW}${message}${NC}"
            ;;
        "error")
            echo -e "${RED}${message}${NC}"
            ;;
        *)
            echo "$message"
            ;;
    esac
}

detect_os() {
    local os
    os="$(uname -s)"

    case "$os" in
        Linux*)
            echo "linux"
            ;;
        Darwin*)
            echo "darwin"
            ;;
        *)
            write_status "Unsupported operating system: $os" "error"
            exit 1
            ;;
    esac
}

detect_arch() {
    local arch
    arch="$(uname -m)"

    case "$arch" in
        x86_64|amd64)
            echo "x64"
            ;;
        aarch64|arm64)
            echo "arm64"
            ;;
        *)
            write_status "Unsupported architecture: $arch" "error"
            exit 1
            ;;
    esac
}

get_download_tool() {
    if command -v curl &> /dev/null; then
        echo "curl"
    elif command -v wget &> /dev/null; then
        echo "wget"
    else
        write_status "Neither curl nor wget found. Please install one of them." "error"
        exit 1
    fi
}

fetch_url() {
    local url="$1"
    local tool
    tool=$(get_download_tool)

    # Use GITHUB_TOKEN for authentication if available (avoids rate limiting)
    if [ "$tool" = "curl" ]; then
        if [ -n "$GITHUB_TOKEN" ]; then
            curl -fsSL -H "Authorization: Bearer $GITHUB_TOKEN" "$url"
        else
            curl -fsSL "$url"
        fi
    else
        if [ -n "$GITHUB_TOKEN" ]; then
            wget -qO- --header="Authorization: Bearer $GITHUB_TOKEN" "$url"
        else
            wget -qO- "$url"
        fi
    fi
}

download_file() {
    local url="$1"
    local output="$2"
    local tool
    tool=$(get_download_tool)

    if [ "$tool" = "curl" ]; then
        curl -fSL "$url" -o "$output"
    else
        wget -q "$url" -O "$output"
    fi
}

get_latest_release() {
    local os="$1"
    local api_url="https://api.github.com/repos/${REPO}/releases/latest"

    write_status "Fetching latest release information..."

    local release_info
    release_info=$(fetch_url "$api_url")

    if [ -z "$release_info" ]; then
        write_status "Failed to fetch release information" "error"
        exit 1
    fi

    # Extract version
    local version
    version=$(echo "$release_info" | grep '"tag_name"' | head -1 | sed 's/.*"tag_name": *"\([^"]*\)".*/\1/')

    # Find the appropriate asset URL based on OS and arch
    local arch="$2"
    local asset_pattern="${os}-${arch}.tar.gz"
    local download_url
    download_url=$(echo "$release_info" | grep '"browser_download_url"' | grep "$asset_pattern" | head -1 | sed 's/.*"browser_download_url": *"\([^"]*\)".*/\1/')

    if [ -z "$download_url" ]; then
        write_status "No release asset found for $os" "error"
        exit 1
    fi

    echo "$version|$download_url"
}

install_libs() {
    local os="$1"
    local release_info="$2"

    local version="${release_info%%|*}"
    local download_url="${release_info#*|}"

    local archive_name
    archive_name=$(basename "$download_url")

    # Check package cache first
    local cache_dir="${HOME}/.sn-cache/downloads"
    local cached_archive="${cache_dir}/${archive_name}"

    if [ -f "$cached_archive" ]; then
        write_status "Using cached ${archive_name}"
    else
        write_status "Downloading ${PKG_NAME} ${version} for ${os}..."
        mkdir -p "$cache_dir"
        if ! download_file "$download_url" "$cached_archive"; then
            write_status "Download failed" "error"
            rm -f "$cached_archive"
            exit 1
        fi
    fi

    local temp_dir
    temp_dir=$(mktemp -d)
    trap "rm -rf '$temp_dir'" EXIT

    write_status "Extracting to ${INSTALL_DIR}..."

    if [ -d "$INSTALL_DIR" ]; then
        rm -rf "$INSTALL_DIR"
    fi
    mkdir -p "$INSTALL_DIR"

    local extract_dir="${temp_dir}/extracted"
    mkdir -p "$extract_dir"
    tar -xzf "$cached_archive" -C "$extract_dir"

    # Handle potentially nested directory structure
    local contents
    contents=$(ls -A "$extract_dir")
    local count
    count=$(echo "$contents" | wc -l)

    if [ "$count" -eq 1 ] && [ -d "${extract_dir}/${contents}" ]; then
        mv "${extract_dir}/${contents}"/* "$INSTALL_DIR/" 2>/dev/null || true
        mv "${extract_dir}/${contents}"/.[!.]* "$INSTALL_DIR/" 2>/dev/null || true
    else
        mv "${extract_dir}"/* "$INSTALL_DIR/" 2>/dev/null || true
        mv "${extract_dir}"/.[!.]* "$INSTALL_DIR/" 2>/dev/null || true
    fi

    write_status "Successfully installed ${PKG_NAME} ${version} to ${INSTALL_DIR}" "success"
}

# Main execution
main() {
    write_status "${PKG_NAME} — native library installer"
    write_status "========================================"

    local os
    os=$(detect_os)
    local arch
    arch=$(detect_arch)
    write_status "Detected OS: ${os} (${arch})"

    # Set install directory based on OS
    INSTALL_DIR="${BASE_DIR}/${os}"

    local release_info
    release_info=$(get_latest_release "$os" "$arch")

    install_libs "$os" "$release_info"

    echo ""
    write_status "Installation complete!" "success"
}

main "$@"
