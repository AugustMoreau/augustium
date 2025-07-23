# Installing Augustium

Augustium can be installed through multiple package managers and methods. Choose the one that works best for your system and workflow.

## Quick Install (Recommended)

### Using NPM (Cross-platform)
```bash
npm install -g augustium
```

### Using Homebrew (macOS/Linux)
```bash
brew install augustium
```

### Using Cargo (Rust users)
```bash
cargo install augustc
```

## Platform-Specific Installation

### macOS

**Option 1: Homebrew (Recommended)**
```bash
brew install augustium
```

**Option 2: NPM**
```bash
npm install -g augustium
```

**Option 3: Direct Download**
1. Download the latest release from [GitHub Releases](https://github.com/AugustMoreau/augustium/releases)
2. Choose `augustium-macos-x86_64.tar.gz` (Intel) or `augustium-macos-aarch64.tar.gz` (Apple Silicon)
3. Extract and add to PATH:
```bash
tar -xzf augustium-macos-*.tar.gz
sudo mv augustc august /usr/local/bin/
```

### Linux

**Option 1: NPM**
```bash
npm install -g augustium
```

**Option 2: Snap Package**
```bash
sudo snap install augustium
```

**Option 3: Direct Download**
1. Download from [GitHub Releases](https://github.com/AugustMoreau/augustium/releases)
2. Choose `augustium-linux-x86_64.tar.gz` or `augustium-linux-aarch64.tar.gz`
3. Extract and install:
```bash
tar -xzf augustium-linux-*.tar.gz
sudo mv augustc august /usr/local/bin/
```

**Option 4: Build from Source**
```bash
git clone https://github.com/AugustMoreau/augustium.git
cd augustium
cargo build --release
sudo cp target/release/{augustc,august} /usr/local/bin/
```

### Windows

**Option 1: NPM**
```bash
npm install -g augustium
```

**Option 2: Chocolatey**
```bash
choco install augustium
```

**Option 3: Direct Download**
1. Download `augustium-windows-x86_64.zip` from [GitHub Releases](https://github.com/AugustMoreau/augustium/releases)
2. Extract to a folder (e.g., `C:\augustium`)
3. Add the folder to your PATH environment variable

**Option 4: Windows Subsystem for Linux (WSL)**
Follow the Linux installation instructions within WSL.

## Container Installation

### Docker
```bash
# Pull the image
docker pull augustium/augustium:latest

# Run interactively
docker run -it augustium/augustium:latest

# Compile a file
docker run -v $(pwd):/workspace augustium/augustium:latest augustc /workspace/hello.aug
```

### Docker Compose
Create a `docker-compose.yml`:
```yaml
version: '3.8'
services:
  augustium:
    image: augustium/augustium:latest
    volumes:
      - .:/workspace
    working_dir: /workspace
    command: august --help
```

## Development Installation

### From Source (Latest)
```bash
git clone https://github.com/AugustMoreau/augustium.git
cd augustium
cargo build --release

# Add to PATH or create symlinks
export PATH="$PWD/target/release:$PATH"
```

### Development Dependencies
To contribute to Augustium development:
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Clone and setup
git clone https://github.com/AugustMoreau/augustium.git
cd augustium
cargo build
cargo test
```

## Verification

After installation, verify that Augustium is working:

```bash
# Check compiler version
augustc --version

# Check CLI tool
august --version

# Create a test project
august new hello-world
cd hello-world
august build
august run
```

## IDE Integration

After installing Augustium, set up your development environment for the best coding experience:

### VS Code (Recommended)

The official Augustium VS Code extension provides comprehensive language support:

#### Installation Methods:

**Method 1: VS Code Marketplace**
1. Open VS Code
2. Press `Ctrl+Shift+X` (or `Cmd+Shift+X` on macOS) to open Extensions
3. Search for "Augustium"
4. Click "Install" on the extension by AugustMoreau

**Method 2: Open VSX Registry**
1. Open VS Code
2. Go to Extensions (`Ctrl+Shift+X`)
3. Search for "Augustium" by AugustMoreau
4. Click "Install"

**Method 3: Command Line**
```bash
# Install from VS Code Marketplace
code --install-extension AugustMoreau.augustium

# Or install manually downloaded .vsix file
code --install-extension augustium-1.0.4.vsix
```

**Method 4: Manual Download**
1. Download the latest `.vsix` file from [GitHub Releases](https://github.com/AugustMoreau/augustium/releases)
2. In VS Code: `Ctrl+Shift+P` → "Extensions: Install from VSIX"
3. Select the downloaded file

#### Extension Features:
- 🎨 **Rich Syntax Highlighting**: Full language support with semantic colors
- 📝 **Smart Code Snippets**: Pre-built templates for:
  - Contract structures
  - Function definitions
  - Common DeFi patterns
  - Event declarations
- 🔍 **IntelliSense**: Auto-completion and intelligent suggestions
- ⚡ **Build Integration**: Compile and run directly from VS Code
- 🐛 **Error Detection**: Real-time syntax and semantic error highlighting
- 📚 **Hover Documentation**: Inline help and documentation
- 🔧 **Code Formatting**: Automatic code formatting and style enforcement

#### Getting Started with VS Code:
1. Install the extension using any method above
2. Create a new `.aug` file or open an existing Augustium project
3. The extension will automatically activate
4. Use `Ctrl+Shift+P` and search "Augustium" to see available commands

### IntelliJ IDEA
1. Install the Augustium plugin from the marketplace
2. Configure the Augustium SDK path

### Vim/Neovim
1. Install the `vim-augustium` plugin
2. Add syntax highlighting configuration

## Troubleshooting

### Common Issues

**Command not found**
- Ensure the installation directory is in your PATH
- Restart your terminal after installation

**Permission denied (macOS/Linux)**
```bash
sudo chmod +x /usr/local/bin/augustc
sudo chmod +x /usr/local/bin/august
```

**NPM installation fails**
- Try with sudo: `sudo npm install -g augustium`
- Or use a Node version manager like nvm

**Cargo installation fails**
- Update Rust: `rustup update`
- Clear cargo cache: `cargo clean`

### Getting Help

- 📖 [Documentation](https://augustium.org/docs)
- 💬 [Discord Community](https://discord.gg/augustium)
- 🐛 [Report Issues](https://github.com/AugustMoreau/augustium/issues)
- 📧 [Email Support](mailto:support@augustium.org)

## Uninstallation

### NPM
```bash
npm uninstall -g augustium
```

### Homebrew
```bash
brew uninstall augustium
```

### Manual
```bash
sudo rm /usr/local/bin/augustc
sudo rm /usr/local/bin/august
```

### Docker
```bash
docker rmi augustium/augustium:latest
```