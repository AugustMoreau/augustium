#!/bin/bash

# Augustium VS Code Extension - Open VSX Publisher
# This script helps publish the Augustium VS Code extension to Open VSX Registry

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
EXTENSION_DIR="vscode-extension"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Functions
print_header() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

check_prerequisites() {
    print_header "Checking Prerequisites"
    
    # Check if ovsx is installed
    if ! command -v ovsx &> /dev/null; then
        print_error "ovsx CLI not found. Installing..."
        npm install -g ovsx
        print_success "ovsx CLI installed"
    else
        print_success "ovsx CLI found"
    fi
    
    # Check if vsce is installed
    if ! command -v vsce &> /dev/null; then
        print_error "vsce CLI not found. Installing..."
        npm install -g @vscode/vsce
        print_success "vsce CLI installed"
    else
        print_success "vsce CLI found"
    fi
    
    # Check for OVSX_PAT environment variable
    if [ -z "$OVSX_PAT" ]; then
        print_warning "OVSX_PAT environment variable not set"
        echo "Please set your Open VSX Personal Access Token:"
        echo "export OVSX_PAT=your-token-here"
        echo ""
        echo "Get your token from: https://open-vsx.org/user-settings/tokens"
        read -p "Enter your Open VSX PAT: " OVSX_PAT
        export OVSX_PAT
    else
        print_success "OVSX_PAT environment variable found"
    fi
}

generate_extension() {
    print_header "Generating VS Code Extension"
    
    cd "$PROJECT_ROOT"
    
    # Check if august CLI is available
    if command -v august &> /dev/null; then
        print_success "Using august CLI to generate extension"
        august generate-plugin vscode --output-dir "$EXTENSION_DIR"
    elif [ -f "scripts/generate-vscode-extension.sh" ]; then
        print_success "Using generation script"
        ./scripts/generate-vscode-extension.sh generate
    else
        print_error "Cannot generate extension. Please ensure august CLI is available or run the generation script manually."
        exit 1
    fi
    
    print_success "Extension generated in $EXTENSION_DIR/"
}

package_extension() {
    print_header "Packaging Extension"
    
    cd "$PROJECT_ROOT/$EXTENSION_DIR"
    
    # Install dependencies if needed
    if [ -f "package.json" ] && [ ! -d "node_modules" ]; then
        print_success "Installing dependencies..."
        npm install
    fi
    
    # Compile TypeScript if needed
    if [ -f "tsconfig.json" ]; then
        print_success "Compiling TypeScript..."
        npm run compile 2>/dev/null || true
    fi
    
    # Package the extension
    print_success "Packaging extension..."
    vsce package
    
    # Find the generated .vsix file
    VSIX_FILE=$(ls *.vsix | head -n 1)
    if [ -z "$VSIX_FILE" ]; then
        print_error "No .vsix file found after packaging"
        exit 1
    fi
    
    print_success "Extension packaged as: $VSIX_FILE"
    echo "$VSIX_FILE" > ../vsix_filename.txt
}

publish_to_openvsx() {
    print_header "Publishing to Open VSX Registry"
    
    cd "$PROJECT_ROOT/$EXTENSION_DIR"
    
    # Get the .vsix filename
    if [ -f "../vsix_filename.txt" ]; then
        VSIX_FILE=$(cat ../vsix_filename.txt)
    else
        VSIX_FILE=$(ls *.vsix | head -n 1)
    fi
    
    if [ -z "$VSIX_FILE" ]; then
        print_error "No .vsix file found. Please package the extension first."
        exit 1
    fi
    
    print_success "Publishing $VSIX_FILE to Open VSX Registry..."
    
    # Publish to Open VSX
    if ovsx publish "$VSIX_FILE" -p "$OVSX_PAT"; then
        print_success "Successfully published to Open VSX Registry!"
        echo ""
        echo "Your extension is now available at:"
        echo "https://open-vsx.org/extension/augustium/augustium"
        echo ""
        echo "Users can install it in VSCodium, Gitpod, and other editors with:"
        echo "codium --install-extension augustium.augustium"
    else
        print_error "Failed to publish to Open VSX Registry"
        exit 1
    fi
}

verify_publication() {
    print_header "Verifying Publication"
    
    print_success "Checking extension on Open VSX Registry..."
    
    # Wait a moment for the registry to update
    sleep 5
    
    if ovsx show augustium.augustium > /dev/null 2>&1; then
        print_success "Extension successfully published and visible on Open VSX!"
        ovsx show augustium.augustium
    else
        print_warning "Extension may still be processing. Check manually at:"
        echo "https://open-vsx.org/extension/augustium/augustium"
    fi
}

cleanup() {
    print_header "Cleanup"
    
    cd "$PROJECT_ROOT"
    
    # Remove temporary files
    rm -f vsix_filename.txt
    
    print_success "Cleanup completed"
}

show_usage() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  full      - Complete process: generate, package, and publish"
    echo "  generate  - Generate VS Code extension only"
    echo "  package   - Package existing extension"
    echo "  publish   - Publish existing .vsix to Open VSX"
    echo "  verify    - Verify extension is published"
    echo "  help      - Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  OVSX_PAT  - Open VSX Personal Access Token (required)"
    echo ""
    echo "Examples:"
    echo "  $0 full                    # Complete publishing process"
    echo "  OVSX_PAT=token $0 publish  # Publish with inline token"
}

# Main script logic
case "${1:-full}" in
    "full")
        print_header "Augustium VS Code Extension - Open VSX Publisher"
        check_prerequisites
        generate_extension
        package_extension
        publish_to_openvsx
        verify_publication
        cleanup
        print_success "All done! Your extension is now available on Open VSX Registry."
        ;;
    "generate")
        check_prerequisites
        generate_extension
        ;;
    "package")
        check_prerequisites
        package_extension
        ;;
    "publish")
        check_prerequisites
        publish_to_openvsx
        verify_publication
        cleanup
        ;;
    "verify")
        verify_publication
        ;;
    "help")
        show_usage
        ;;
    *)
        print_error "Unknown command: $1"
        show_usage
        exit 1
        ;;
esac