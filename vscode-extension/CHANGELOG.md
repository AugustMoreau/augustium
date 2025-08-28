# Change Log

All notable changes to the Augustium VS Code extension will be documented in this file.

## [1.2.0] - 2024-01-XX

### Added
- **Machine Learning Support**: Complete syntax highlighting for ML keywords and functions
- **ML Code Snippets**: Pre-built templates for neural networks, training loops, and tensor operations
- **Computer Vision Snippets**: CNN model templates and image processing functions
- **Enhanced Type Support**: Added f32, f64 floating-point types for ML computations
- **ML Keywords**: Tensor, Model, Layer, Optimizer, Dataset, GPU, CUDA operations
- **ML Functions**: train, predict, forward, backward, compile, fit, evaluate
- **ML Types**: Conv2D, LSTM, Transformer, ResNet, VGG, BERT, GPT, ReLU, Softmax
- **Tensor Operations**: zeros, ones, randn, eye, matmul, relu, sigmoid, tanh, softmax

### Changed
- **Extension Description**: Updated to highlight ML capabilities
- **Version Bump**: Updated to 1.2.0 for ML feature release

### Technical
- Enhanced augustium.tmLanguage.json with ML-specific syntax patterns
- Added comprehensive ML code snippets in augustium.json
- Updated package.json description to reflect ML capabilities

## [1.1.0] - 2024-01-XX

### Added
- **Language Server Protocol Integration**: Full LSP client implementation
- **Real-time Error Detection**: Live error checking and diagnostics
- **Code Completion**: Context-aware autocompletion for Augustium syntax
- **Hover Information**: Documentation and type information on hover
- **Configuration Options**: LSP path and tracing configuration
- **Restart Command**: Command to restart the language server
- **Enhanced Error Messages**: Better error reporting with actionable suggestions

### Changed
- **Extension Description**: Updated to reflect LSP capabilities
- **Version Bump**: Updated to 1.1.0 for major feature release
- **Dependencies**: Added vscode-languageclient for LSP support

### Technical
- Complete rewrite of extension.ts with LSP client
- Added TypeScript compilation step
- Enhanced package.json with LSP configuration
- Added publishing scripts for VS Code Marketplace and Open VSX

## [1.0.4] - 2024-01-XX

### Added
- Basic syntax highlighting for `.aug` files
- Code snippets for common Augustium patterns
- Language configuration for bracket matching and commenting

### Features
- Syntax highlighting for keywords, strings, comments
- Basic language support without LSP
- Snippet support for contracts, functions, and events

---

**Note**: This extension requires the Augustium compiler to be installed (`npm install -g augustium`) for full LSP functionality.