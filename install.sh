#!/bin/bash
# Augustium Installation Script
# Usage: curl -sSf https://augustium.org/install.sh | sh

set -e

# Configuration
GITHUB_REPO="AugustMoreau/augustium"
INSTALL_DIR="$HOME/.augustium"
BIN_DIR="$HOME/.local/bin"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
    exit 1
}

# Detect platform and architecture
detect_platform() {
    local os="$(uname -s)"
    local arch="$(uname -m)"
    
    case "$os" in
        Linux*)
            PLATFORM="linux"
            ;;
        Darwin*)
            PLATFORM="macos"
            ;;
        CYGWIN*|MINGW*|MSYS*)
            PLATFORM="windows"
            ;;
        *)
            error "Unsupported operating system: $os"
            ;;
    esac
    
    case "$arch" in
        x86_64|amd64)
            ARCH="x86_64"
            ;;
        aarch64|arm64)
            ARCH="aarch64"
            ;;
        *)
            error "Unsupported architecture: $arch"
            ;;
    esac
    
    PLATFORM_ARCH="${PLATFORM}-${ARCH}"
    info "Detected platform: $PLATFORM_ARCH"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Download file
download() {
    local url="$1"
    local output="$2"
    
    if command_exists curl; then
        curl -sSL "$url" -o "$output"
    elif command_exists wget; then
        wget -q "$url" -O "$output"
    else
        error "Neither curl nor wget is available. Please install one of them."
    fi
}

# Get latest release info
get_latest_release() {
    local api_url="https://api.github.com/repos/$GITHUB_REPO/releases/latest"
    local temp_file="$(mktemp)"
    
    info "Fetching latest release information..."
    download "$api_url" "$temp_file"
    
    # Extract tag name and download URL
    if command_exists jq; then
        TAG_NAME=$(jq -r '.tag_name' "$temp_file")
        DOWNLOAD_URL=$(jq -r ".assets[] | select(.name | contains(\"$PLATFORM_ARCH\")) | .browser_download_url" "$temp_file")
    else
        # Fallback parsing without jq
        TAG_NAME=$(grep '"tag_name"' "$temp_file" | sed 's/.*"tag_name":\s*"\([^"]*\)".*/\1/')
        DOWNLOAD_URL=$(grep "$PLATFORM_ARCH" "$temp_file" | grep 'browser_download_url' | sed 's/.*"browser_download_url":\s*"\([^"]*\)".*/\1/')
    fi
    
    rm -f "$temp_file"
    
    if [ -z "$DOWNLOAD_URL" ]; then
        error "No pre-built binary found for $PLATFORM_ARCH. Please build from source."
    fi
    
    info "Latest version: $TAG_NAME"
}

# Install Augustium
install_augustium() {
    info "Installing Augustium..."
    
    # Create directories
    mkdir -p "$INSTALL_DIR" "$BIN_DIR"
    
    # Download archive
    local archive_name="augustium-${PLATFORM_ARCH}.tar.gz"
    if [ "$PLATFORM" = "windows" ]; then
        archive_name="augustium-${PLATFORM_ARCH}.zip"
    fi
    
    local archive_path="$INSTALL_DIR/$archive_name"
    
    info "Downloading $archive_name..."
    download "$DOWNLOAD_URL" "$archive_path"
    
    # Extract archive
    info "Extracting archive..."
    cd "$INSTALL_DIR"
    
    if [ "$PLATFORM" = "windows" ]; then
        if command_exists unzip; then
            unzip -q "$archive_path"
        else
            error "unzip command not found. Please install unzip."
        fi
    else
        tar -xzf "$archive_path"
    fi
    
    # Move binaries
    local ext=""
    if [ "$PLATFORM" = "windows" ]; then
        ext=".exe"
    fi
    
    if [ -f "augustc$ext" ]; then
        mv "augustc$ext" "$BIN_DIR/"
        chmod +x "$BIN_DIR/augustc$ext"
        success "Installed augustc"
    fi
    
    if [ -f "august$ext" ]; then
        mv "august$ext" "$BIN_DIR/"
        chmod +x "$BIN_DIR/august$ext"
        success "Installed august"
    fi
    
    # Cleanup
    rm -f "$archive_path"
    
    success "Augustium installed successfully!"
}

# Update PATH
update_path() {
    local shell_profile=""
    
    # Detect shell and profile file
    if [ -n "$BASH_VERSION" ]; then
        shell_profile="$HOME/.bashrc"
    elif [ -n "$ZSH_VERSION" ]; then
        shell_profile="$HOME/.zshrc"
    elif [ "$SHELL" = "/bin/bash" ]; then
        shell_profile="$HOME/.bashrc"
    elif [ "$SHELL" = "/bin/zsh" ] || [ "$SHELL" = "/usr/bin/zsh" ]; then
        shell_profile="$HOME/.zshrc"
    else
        shell_profile="$HOME/.profile"
    fi
    
    # Check if PATH already contains our bin directory
    if echo "$PATH" | grep -q "$BIN_DIR"; then
        info "$BIN_DIR is already in PATH"
        return
    fi
    
    # Add to PATH in profile
    if [ -f "$shell_profile" ]; then
        if ! grep -q "$BIN_DIR" "$shell_profile"; then
            echo "" >> "$shell_profile"
            echo "# Augustium" >> "$shell_profile"
            echo "export PATH=\"$BIN_DIR:\$PATH\"" >> "$shell_profile"
            success "Added $BIN_DIR to PATH in $shell_profile"
        fi
    fi
    
    # Export for current session
    export PATH="$BIN_DIR:$PATH"
}

# Verify installation
verify_installation() {
    info "Verifying installation..."
    
    if command_exists augustc; then
        local version=$(augustc --version 2>/dev/null || echo "unknown")
        success "augustc is working: $version"
    else
        warn "augustc not found in PATH. You may need to restart your shell."
    fi
    
    if command_exists august; then
        local version=$(august --version 2>/dev/null || echo "unknown")
        success "august is working: $version"
    else
        warn "august not found in PATH. You may need to restart your shell."
    fi
}

# Show next steps
show_next_steps() {
    echo ""
    echo "🎉 Augustium has been installed!"
    echo ""
    echo "To get started:"
    echo "  1. Restart your shell or run: source ~/.bashrc (or ~/.zshrc)"
    echo "  2. Verify installation: augustc --version"
    echo "  3. Create a new project: august new my-project"
    echo "  4. Read the documentation: https://docs.augustium.org"
    echo ""
    echo "If you encounter any issues:"
    echo "  - Check our FAQ: https://augustium.org/faq"
    echo "  - Join our Discord: https://discord.gg/augustium"
    echo "  - Report bugs: https://github.com/$GITHUB_REPO/issues"
    echo ""
}

# Main installation flow
main() {
    echo "🚀 Augustium Installation Script"
    echo "================================"
    echo ""
    
    # Check for help flag
    if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --help, -h     Show this help message"
        echo "  --uninstall    Uninstall Augustium"
        echo ""
        echo "Environment variables:"
        echo "  AUGUSTIUM_INSTALL_DIR    Installation directory (default: ~/.augustium)"
        echo "  AUGUSTIUM_BIN_DIR        Binary directory (default: ~/.local/bin)"
        echo ""
        exit 0
    fi
    
    # Check for uninstall flag
    if [ "$1" = "--uninstall" ]; then
        info "Uninstalling Augustium..."
        rm -rf "$INSTALL_DIR"
        rm -f "$BIN_DIR/augustc" "$BIN_DIR/august"
        success "Augustium uninstalled successfully!"
        exit 0
    fi
    
    # Override directories if environment variables are set
    if [ -n "$AUGUSTIUM_INSTALL_DIR" ]; then
        INSTALL_DIR="$AUGUSTIUM_INSTALL_DIR"
    fi
    
    if [ -n "$AUGUSTIUM_BIN_DIR" ]; then
        BIN_DIR="$AUGUSTIUM_BIN_DIR"
    fi
    
    # Run installation steps
    detect_platform
    get_latest_release
    install_augustium
    update_path
    verify_installation
    show_next_steps
}

# Run main function
main "$@"