# Augustium VS Code Extension

Official VS Code extension for the Augustium smart contract programming language.

## Features

- **Syntax Highlighting**: Full syntax highlighting for `.aug` files
- **Language Server Protocol**: Advanced IDE features powered by the Augustium LSP
- **Error Detection**: Real-time error detection and reporting
- **Code Completion**: Context-aware autocompletion
- **Hover Information**: Documentation on hover
- **Code Snippets**: Pre-built snippets for common patterns
- **Diagnostics**: Advanced error reporting with precise locations

## Installation

### Prerequisites

Make sure you have Augustium installed:

```bash
npm install -g augustium
```

### Install Extension

1. **From VS Code Marketplace**: Search for "Augustium" in the Extensions panel
2. **From VSIX**: Download the `.vsix` file and install via `code --install-extension augustium-1.1.0.vsix`

## Configuration

The extension can be configured through VS Code settings:

- `augustium.lsp.path`: Path to the Augustium LSP server executable (default: `augustium-lsp`)
- `augustium.lsp.trace.server`: LSP communication tracing level (`off`, `messages`, `verbose`)

## Commands

- `Augustium: Restart Language Server`: Restart the LSP server if it becomes unresponsive

## Usage

1. Create a new file with `.aug` extension
2. Start writing Augustium smart contracts
3. Enjoy syntax highlighting, error detection, and autocompletion

## Example

```augustium
use std::collections::HashMap;
use std::events::Event;

#[derive(Event)]
struct Transfer {
    from: address,
    to: address,
    value: u256
}

contract SimpleToken {
    name: string;
    symbol: string;
    total_supply: u256;
    balances: HashMap<address, u256>;
    
    constructor(name: string, symbol: string, supply: u256) {
        self.name = name;
        self.symbol = symbol;
        self.total_supply = supply;
        self.balances.insert(msg.sender, supply);
    }
}
```

## Troubleshooting

### Language Server Not Starting

If you see "Failed to start Augustium Language Server" error:

1. Make sure `augustium` is installed: `npm install -g augustium`
2. Verify `augustium-lsp` is in your PATH: `augustium-lsp --version`
3. Check the Output panel (View → Output → Augustium Language Server) for detailed error messages
4. Try restarting the language server: `Ctrl+Shift+P` → "Augustium: Restart Language Server"

### Custom LSP Path

If you have a custom installation, set the LSP path in settings:

```json
{
    "augustium.lsp.path": "/path/to/your/augustium-lsp"
}
```

## Resources

### Documentation
- [Language Reference](https://github.com/AugustMoreau/augustium/blob/main/docs/API_REFERENCE.md)
- [Quick Start Guide](https://github.com/AugustMoreau/augustium/blob/main/docs/QUICK_START.md)
- [Examples](https://github.com/AugustMoreau/augustium/tree/main/examples)

### Community
- [GitHub Issues](https://github.com/AugustMoreau/augustium/issues) - Report bugs or request features
- [Discussions](https://github.com/AugustMoreau/augustium/discussions) - Ask questions and share ideas

## FAQ

### Q: Why isn't syntax highlighting working?
A: Make sure your file has the `.aug` extension. The extension only activates for Augustium files.

### Q: Can I use this extension with other editors?
A: This extension is specifically for VS Code. However, the Augustium LSP server can be integrated with other LSP-compatible editors like Neovim, Emacs, or Sublime Text.

### Q: How do I enable debug logging?
A: Set `"augustium.lsp.trace.server": "verbose"` in your VS Code settings to see detailed LSP communication logs.

### Q: The extension seems slow, what can I do?
A: Try restarting the language server using the command palette: `Ctrl+Shift+P` → "Augustium: Restart Language Server"

## Version History

See [CHANGELOG.md](CHANGELOG.md) for detailed version history and feature updates.

## Support

If you encounter issues:
1. Check the [Troubleshooting](#troubleshooting) section
2. Look at existing [GitHub Issues](https://github.com/AugustMoreau/augustium/issues)
3. Create a new issue with detailed information about your problem