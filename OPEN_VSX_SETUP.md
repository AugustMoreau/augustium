# Publishing Augustium VS Code Extension to Open VSX Registry

This guide explains how to publish the Augustium VS Code extension to the Open VSX Registry, making it available for VS Code alternatives like VSCodium, Gitpod, Eclipse Theia, and other editors that use the Open VSX marketplace.

## What is Open VSX?

Open VSX Registry is an open-source alternative to the VS Code Marketplace, hosted by the Eclipse Foundation. It provides extensions for:

- **VSCodium** - Open-source VS Code without Microsoft branding/telemetry
- **Gitpod** - Cloud development environments
- **Eclipse Theia** - Cloud & desktop IDE platform
- **Code - OSS** - Open-source VS Code builds
- **Other editors** that implement the Language Server Protocol

## Prerequisites

### 1. Install Required Tools

```bash
# Install ovsx CLI tool
npm install -g ovsx

# Verify installation
ovsx --version
```

### 2. Create Open VSX Account

1. Go to [open-vsx.org](https://open-vsx.org)
2. Sign in with GitHub account
3. Go to your profile settings
4. Generate a Personal Access Token (PAT)
5. Save the token securely

### 3. Set Up Environment Variables

```bash
# Add to your shell profile (.bashrc, .zshrc, etc.)
export OVSX_PAT="your-open-vsx-personal-access-token"

# Or create a .env file in the project root
echo "OVSX_PAT=your-open-vsx-personal-access-token" >> .env
```

## Manual Publishing Process

### Step 1: Generate the Extension

```bash
# Generate VS Code extension files
./scripts/generate-vscode-extension.sh generate

# Or if august CLI is available
august ide vscode
```

### Step 2: Navigate to Extension Directory

```bash
cd vscode-extension
```

### Step 3: Install Dependencies

```bash
npm install
```

### Step 4: Package the Extension

```bash
# Install vsce if not already installed
npm install -g vsce

# Package the extension
vsce package
```

This creates a `.vsix` file (e.g., `augustium-0.1.0.vsix`)

### Step 5: Publish to Open VSX

```bash
# Publish using ovsx CLI
ovsx publish augustium-0.1.0.vsix -p $OVSX_PAT

# Or specify the registry URL explicitly
ovsx publish augustium-0.1.0.vsix -p $OVSX_PAT --registryUrl https://open-vsx.org
```

### Step 6: Verify Publication

1. Visit [open-vsx.org](https://open-vsx.org)
2. Search for "Augustium"
3. Verify the extension appears with correct metadata

## Automated Publishing with GitHub Actions

The project includes a GitHub Actions workflow that automatically publishes to Open VSX on releases.

### Required GitHub Secrets

Add these secrets to your GitHub repository:

1. Go to **Settings** → **Secrets and variables** → **Actions**
2. Add the following secrets:

```
OVSX_PAT=your-open-vsx-personal-access-token
VSCE_PAT=your-vscode-marketplace-token (optional)
```

### Workflow Triggers

The workflow runs automatically when:

- A new release is created
- Manual workflow dispatch is triggered
- Push to main branch (for testing)

### Manual Workflow Trigger

1. Go to **Actions** tab in GitHub
2. Select "Publish VS Code Extension"
3. Click "Run workflow"
4. Choose branch and click "Run workflow"

## Extension Configuration

### Package.json Structure

The extension's `package.json` includes:

```json
{
  "name": "augustium",
  "displayName": "Augustium Language Support",
  "description": "Comprehensive language support for Augustium",
  "version": "0.1.0",
  "publisher": "augustium",
  "engines": {
    "vscode": "^1.60.0"
  },
  "categories": [
    "Programming Languages",
    "Snippets"
  ],
  "contributes": {
    "languages": [...],
    "grammars": [...],
    "snippets": [...]
  }
}
```

### Open VSX Specific Configuration

The `.ovsx.json` file contains Open VSX specific metadata:

- Registry URL and authentication
- Extension metadata and badges
- File paths and dependencies
- Contribution points and capabilities

## Extension Features

The Augustium VS Code extension provides:

### Language Support
- **Syntax Highlighting** - Full syntax highlighting for `.aug` files
- **Language Configuration** - Bracket matching, auto-indentation
- **File Association** - Automatic detection of Augustium files

### Code Assistance
- **Snippets** - Common Augustium code patterns
- **Auto-completion** - Basic keyword completion
- **Bracket Matching** - Automatic bracket pairing

### Build Integration
- **Compiler Integration** - Run `augustc` from VS Code
- **CLI Integration** - Access `august` commands
- **Error Highlighting** - Display compilation errors

### Configuration Options
- `augustium.compiler.path` - Path to augustc compiler
- `augustium.cli.path` - Path to august CLI
- `augustium.format.onSave` - Auto-format on save
- `augustium.linting.enabled` - Enable/disable linting

## User Installation

Once published to Open VSX, users can install the extension in:

### VSCodium
```bash
# Command line installation
codium --install-extension augustium.augustium

# Or through Extensions view: Search "Augustium"
```

### Gitpod
```yaml
# .gitpod.yml
vscode:
  extensions:
    - augustium.augustium
```

### Eclipse Theia
- Open Extensions view
- Search for "Augustium"
- Click Install

## Updating the Extension

### Version Management

1. Update version in `package.json`
2. Update `CHANGELOG.md` with new features
3. Commit changes
4. Create new GitHub release
5. Automated workflow publishes to Open VSX

### Manual Update

```bash
# Increment version
npm version patch  # or minor/major

# Package new version
vsce package

# Publish to Open VSX
ovsx publish augustium-x.y.z.vsix -p $OVSX_PAT
```

## Troubleshooting

### Common Issues

**Authentication Error**
```
Error: Request failed with status code 401
```
- Verify OVSX_PAT is correct
- Check token hasn't expired
- Ensure token has publish permissions

**Package Validation Error**
```
Error: Extension validation failed
```
- Check package.json syntax
- Verify all required fields are present
- Ensure file paths in contributes section exist

**File Not Found Error**
```
Error: ENOENT: no such file or directory
```
- Verify all referenced files exist
- Check grammar, snippet, and icon file paths
- Ensure language-configuration.json exists

### Debug Commands

```bash
# Validate package without publishing
ovsx verify augustium-0.1.0.vsix

# Check extension info
ovsx show augustium.augustium

# List published versions
ovsx search augustium
```

## Comparison: Open VSX vs VS Code Marketplace

| Feature | Open VSX | VS Code Marketplace |
|---------|----------|--------------------|
| **License** | Open source | Proprietary |
| **Hosting** | Eclipse Foundation | Microsoft |
| **Editors** | VSCodium, Theia, Gitpod | VS Code |
| **Publishing** | Free | Free |
| **Review Process** | Automated | Manual + Automated |
| **API Access** | Open API | Restricted API |
| **Telemetry** | Minimal | Microsoft telemetry |

## Best Practices

### Extension Quality
- Include comprehensive README
- Add meaningful keywords and categories
- Provide clear installation instructions
- Include screenshots and examples
- Maintain changelog

### Security
- Never commit PAT tokens to repository
- Use GitHub secrets for CI/CD
- Regularly rotate access tokens
- Review extension permissions

### Maintenance
- Monitor extension usage statistics
- Respond to user issues promptly
- Keep dependencies updated
- Test with multiple editors

## Resources

- [Open VSX Registry](https://open-vsx.org)
- [ovsx CLI Documentation](https://github.com/eclipse/openvsx/tree/master/cli)
- [VS Code Extension API](https://code.visualstudio.com/api)
- [VSCodium](https://vscodium.com)
- [Eclipse Theia](https://theia-ide.org)
- [Gitpod Documentation](https://www.gitpod.io/docs)

## Support

For issues with:
- **Extension functionality**: [GitHub Issues](https://github.com/AugustMoreau/augustium/issues)
- **Open VSX publishing**: [Open VSX GitHub](https://github.com/eclipse/openvsx/issues)
- **VSCodium**: [VSCodium GitHub](https://github.com/VSCodium/vscodium/issues)