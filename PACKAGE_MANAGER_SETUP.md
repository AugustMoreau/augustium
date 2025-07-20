# Package Manager Setup Guide

This guide explains how to set up and publish Augustium to various package managers, enabling users to install it with simple commands like `npm install augustium` or `brew install augustium`.

## Overview

We've configured Augustium for distribution through:

1. **Cargo** (Rust ecosystem) - `cargo install augustc`
2. **NPM** (JavaScript ecosystem) - `npm install -g augustium`
3. **Homebrew** (macOS/Linux) - `brew install augustium`
4. **Docker Hub** (Containers) - `docker pull augustium/augustium`
5. **GitHub Releases** (Direct downloads)
6. **Shell Script** (One-liner install) - `curl -sSf https://augustium.org/install.sh | sh`

## Prerequisites

Before publishing, ensure you have:

- [ ] GitHub repository with proper releases
- [ ] CI/CD pipeline (GitHub Actions) configured
- [ ] Accounts on package manager platforms
- [ ] API tokens/credentials for automated publishing

## 1. Cargo (crates.io)

### Setup
1. Create account at [crates.io](https://crates.io)
2. Generate API token: `cargo login`
3. Update `Cargo.toml` with proper metadata (already done)

### Publishing
```bash
# Manual publish
cargo publish

# Automated via GitHub Actions
# Set CARGO_REGISTRY_TOKEN secret in repository
```

### User Installation
```bash
cargo install augustc
```

## 2. NPM (npmjs.com)

### Setup
1. Create account at [npmjs.com](https://npmjs.com)
2. Generate access token with publish permissions
3. Set `NPM_TOKEN` secret in GitHub repository

### Files Created
- `npm-package.json` - NPM package configuration
- `scripts/install.js` - Installation script that downloads binaries

### Publishing
```bash
# Manual publish
cp npm-package.json package.json
npm publish

# Automated via GitHub Actions
# Publishes on every release
```

### User Installation
```bash
npm install -g augustium
```

## 3. Homebrew

### Setup
1. Create a Homebrew tap repository: `homebrew-augustium`
2. Add the formula file to the tap
3. Set up automated updates

### Files Created
- `Formula/augustium.rb` - Homebrew formula

### Publishing
```bash
# Create tap repository
gh repo create AugustMoreau/homebrew-augustium --public

# Add formula
cp Formula/augustium.rb /path/to/homebrew-augustium/Formula/
cd /path/to/homebrew-augustium
git add Formula/augustium.rb
git commit -m "Add augustium formula"
git push

# Automated updates via GitHub Actions
# Set HOMEBREW_TOKEN secret
```

### User Installation
```bash
# Add tap
brew tap AugustMoreau/augustium

# Install
brew install augustium
```

## 4. Docker Hub

### Setup
1. Create account at [Docker Hub](https://hub.docker.com)
2. Create repository: `augustium/augustium`
3. Generate access token
4. Set `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` secrets

### Files Created
- `Dockerfile` - Multi-stage build configuration

### Publishing
```bash
# Manual publish
docker build -t augustium/augustium:latest .
docker push augustium/augustium:latest

# Automated via GitHub Actions
# Builds and pushes on every release
```

### User Installation
```bash
# Pull and run
docker pull augustium/augustium:latest
docker run -it augustium/augustium:latest

# Use in projects
docker run -v $(pwd):/workspace augustium/augustium:latest augustc /workspace/contract.aug
```

## 5. GitHub Releases

### Setup
- Automated via GitHub Actions
- Creates cross-platform binaries
- Uploads to release assets

### Files Created
- `.github/workflows/release.yml` - CI/CD pipeline

### Publishing
```bash
# Create a release
gh release create v0.1.0 --title "Augustium v0.1.0" --notes "Initial release"

# Automated builds will:
# 1. Build for multiple platforms
# 2. Create archives
# 3. Upload to release
```

### User Installation
```bash
# Download from releases page
wget https://github.com/AugustMoreau/augustium/releases/download/v0.1.0/augustium-linux-x86_64.tar.gz
tar -xzf augustium-linux-x86_64.tar.gz
sudo mv augustc august /usr/local/bin/
```

## 6. Shell Script Installer

### Setup
- Host `install.sh` on your website
- Ensure HTTPS for security

### Files Created
- `install.sh` - Universal installation script

### Publishing
```bash
# Host on your website
cp install.sh /path/to/website/install.sh

# Or use GitHub Pages
# The script will be available at:
# https://augustium.org/install.sh
```

### User Installation
```bash
# One-liner install
curl -sSf https://augustium.org/install.sh | sh

# Or download and inspect first
curl -sSf https://augustium.org/install.sh -o install.sh
chmod +x install.sh
./install.sh
```

## Required Secrets

Add these secrets to your GitHub repository:

```bash
# Cargo
CARGO_REGISTRY_TOKEN=your_crates_io_token

# NPM
NPM_TOKEN=your_npm_token

# Docker Hub
DOCKERHUB_USERNAME=your_dockerhub_username
DOCKERHUB_TOKEN=your_dockerhub_token

# Homebrew (optional, for automated updates)
HOMEBREW_TOKEN=your_github_token_with_repo_access
```

## Testing

Before publishing, test each installation method:

```bash
# Test Cargo build
cargo build --release

# Test NPM package
node scripts/install.js

# Test Docker build
docker build -t augustium-test .

# Test Homebrew formula
brew install --build-from-source ./Formula/augustium.rb

# Test shell installer
bash install.sh
```

## Release Process

1. **Prepare Release**
   - Update version in `Cargo.toml`
   - Update version in `npm-package.json`
   - Update CHANGELOG.md
   - Test all builds locally

2. **Create Release**
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   gh release create v0.1.0 --generate-notes
   ```

3. **Automated Publishing**
   - GitHub Actions will automatically:
     - Build cross-platform binaries
     - Publish to Cargo
     - Publish to NPM
     - Push to Docker Hub
     - Update Homebrew formula

4. **Manual Steps**
   - Update website with new version
   - Announce on social media
   - Update documentation

## Monitoring

Track installation metrics:

- **Cargo**: [crates.io stats](https://crates.io/crates/augustc)
- **NPM**: [npmjs.com stats](https://npmjs.com/package/augustium)
- **Docker**: Docker Hub download stats
- **GitHub**: Release download stats
- **Homebrew**: Analytics via tap repository

## Troubleshooting

### Common Issues

1. **Build Failures**
   - Check Rust version compatibility
   - Verify all dependencies are available
   - Test cross-compilation locally

2. **Publishing Failures**
   - Verify API tokens are valid
   - Check package name availability
   - Ensure version numbers are incremented

3. **Installation Issues**
   - Test on clean systems
   - Verify binary permissions
   - Check PATH configuration

### Getting Help

- Check GitHub Actions logs for CI/CD issues
- Test locally before pushing
- Use package manager documentation
- Ask community for help

## Future Enhancements

- **Chocolatey** (Windows package manager)
- **Snap** (Linux universal packages)
- **Flatpak** (Linux application distribution)
- **MacPorts** (macOS package manager)
- **Arch AUR** (Arch Linux user repository)
- **Debian/Ubuntu PPA** (Personal Package Archive)