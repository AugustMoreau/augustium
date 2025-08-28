# Augustium WASM Integration Layer

This package provides WebAssembly bindings for the Augustium smart contract platform, enabling seamless integration of Augustium functionality into web applications.

## Features

- **Complete AVM Integration**: Full WebAssembly bindings that mirror AVM functionality
- **State Compatibility**: Seamless state management between AVM and WASM execution
- **JavaScript APIs**: High-level JavaScript APIs for easy web integration
- **TypeScript Support**: Complete TypeScript definitions for type safety
- **Event System**: Real-time event handling for contract interactions
- **Utility Functions**: Helper functions for address generation, validation, and more

## Installation

### Building from Source

```bash
# Install wasm-pack if you haven't already
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Build the WASM package
wasm-pack build --target web --out-dir pkg

# For Node.js target
wasm-pack build --target nodejs --out-dir pkg-node
```

### Using in Your Project

#### Web (ES Modules)

```javascript
import init, { AugustiumRuntime, WebUtils } from './path/to/augustium_wasm.js';

async function main() {
    // Initialize the WASM module
    await init();
    
    // Create runtime instance
    const runtime = new AugustiumRuntime();
    
    // Your code here...
}
```

#### Node.js

```javascript
const { AugustiumRuntime, WebUtils } = require('./path/to/augustium_wasm');

const runtime = new AugustiumRuntime();
// Your code here...
```

## Quick Start

### Basic Contract Deployment

```javascript
import init, { AugustiumRuntime, AugustiumConfig, WebUtils } from 'augustium-wasm';

async function deployContract() {
    await init();
    
    // Configure runtime
    const config = new AugustiumConfig();
    config.gasLimit = 2000000;
    config.debugMode = true;
    
    const runtime = new AugustiumRuntime(config);
    
    // Generate contract address
    const address = WebUtils.generateAddress();
    
    // Deploy contract from source
    const contractSource = `
        contract SimpleStorage {
            uint256 private value;
            
            function setValue(uint256 newValue) public {
                value = newValue;
            }
            
            function getValue() public view returns (uint256) {
                return value;
            }
        }
    `;
    
    try {
        const result = await runtime.deployContractFromSource(contractSource, address);
        console.log('Contract deployed:', result);
        return address;
    } catch (error) {
        console.error('Deployment failed:', error);
    }
}
```

### Contract Interaction

```javascript
import { ContractHelper } from 'augustium-wasm';

async function interactWithContract(runtime, contractAddress) {
    const contract = new ContractHelper(runtime, contractAddress);
    
    // Call contract method
    const result = await contract.call('setValue', { newValue: 42 });
    console.log('Method call result:', result);
    
    // Get contract balance
    const balance = contract.balance();
    console.log('Contract balance:', balance);
}
```

### Event Handling

```javascript
function setupEventListeners(runtime) {
    // Listen for transfer events
    runtime.on('transfer', (eventData) => {
        console.log('Transfer:', {
            from: eventData.from,
            to: eventData.to,
            amount: eventData.amount
        });
    });
    
    // Listen for custom contract events
    runtime.on('ValueChanged', (eventData) => {
        console.log('Value changed:', eventData);
    });
}
```

### State Management

```javascript
async function manageState(runtime) {
    // Export current state
    const state = runtime.exportState();
    
    // Save to localStorage (browser) or file (Node.js)
    localStorage.setItem('augustium-state', JSON.stringify(state));
    
    // Later, restore state
    const savedState = JSON.parse(localStorage.getItem('augustium-state'));
    runtime.importState(savedState);
}
```

## API Reference

### Core Classes

#### `AugustiumRuntime`

The main runtime class for executing Augustium contracts.

```typescript
class AugustiumRuntime {
    constructor(config?: AugustiumConfig);
    
    // Contract management
    deployContractFromSource(source: string, address: string): Promise<string>;
    deployContract(bytecode: Uint8Array, address: string): void;
    callContract(address: string, method: string, args: any): Promise<any>;
    
    // Execution
    executeBytecode(bytecode: Uint8Array): Promise<any>;
    
    // Balance and transfers
    getBalance(address: string): number;
    transfer(from: string, to: string, amount: number): void;
    
    // Events
    on(event: string, callback: (data: any) => void): void;
    off(event: string): void;
    getEvents(): Event[];
    clearEvents(): void;
    
    // State management
    getStats(): RuntimeStats;
    reset(): void;
    exportState(): StateExport;
    importState(stateData: StateExport): void;
}
```

#### `AugustiumConfig`

Configuration options for the runtime.

```typescript
class AugustiumConfig {
    gasLimit: number;        // Default: 1,000,000
    debugMode: boolean;      // Default: false
    enableEvents: boolean;   // Default: true
    maxStackSize: number;    // Default: 1024
    networkId: number;       // Default: 1
}
```

#### `ContractHelper`

Helper class for easier contract interaction.

```typescript
class ContractHelper {
    constructor(runtime: AugustiumRuntime, address: string);
    
    call(method: string, args: any): Promise<ExecutionResult>;
    readonly address: string;
    balance(): number;
}
```

#### `WebUtils`

Utility functions for web development.

```typescript
class WebUtils {
    static generateAddress(): string;
    static validateAddress(address: string): boolean;
    static toHex(value: number): string;
    static fromHex(hexStr: string): number;
    static now(): number;
    static createTxHash(from: string, to: string, value: number, nonce: number): string;
}
```

### Low-Level WASM Bindings

#### `WasmAvm`

Direct WASM bindings to the AVM.

```typescript
class WasmAvm {
    constructor();
    
    executeBytecode(bytecode: Uint8Array): Promise<any>;
    executeInstruction(opcode: number, operands: Uint8Array): Promise<any>;
    
    // Stack operations
    pushStack(value: any): void;
    popStack(): any;
    peekStack(): any;
    getStackSize(): number;
    clearStack(): void;
    
    // Contract operations
    deployContract(bytecode: Uint8Array, address: Uint8Array): Promise<void>;
    callContract(address: Uint8Array, method: string, args: any): Promise<any>;
    
    // State management
    getState(): Promise<any>;
    setState(state: any): Promise<void>;
    
    // Utilities
    setDebugMode(enabled: boolean): void;
    getGasUsed(): number;
    reset(): void;
}
```

## Examples

See the `examples/` directory for comprehensive examples:

- `web_integration.js` - Complete web integration examples
- `node_example.js` - Node.js usage examples
- `react_component.jsx` - React component integration
- `vue_component.vue` - Vue.js component integration

## Error Handling

The WASM bindings provide comprehensive error handling:

```javascript
try {
    const result = await runtime.callContract(address, 'method', args);
} catch (error) {
    if (error instanceof CompilationError) {
        console.error('Compilation failed:', error.message, 'at line', error.line);
    } else if (error instanceof RuntimeError) {
        console.error('Runtime error:', error.message, 'gas used:', error.gasUsed);
    } else if (error instanceof ContractError) {
        console.error('Contract error:', error.message, 'at address:', error.address);
    } else {
        console.error('Unknown error:', error);
    }
}
```

## Performance Considerations

### Gas Optimization

```javascript
// Set appropriate gas limits
const config = new AugustiumConfig();
config.gasLimit = 1000000; // Adjust based on your needs

// Monitor gas usage
const stats = runtime.getStats();
console.log('Gas used:', stats.gasUsed);
```

### Memory Management

```javascript
// Clear events periodically to free memory
runtime.clearEvents();

// Reset runtime when done
runtime.reset();
```

### Batch Operations

```javascript
// Batch multiple operations for better performance
const operations = [
    { type: 'call', address: addr1, method: 'method1', args: {} },
    { type: 'call', address: addr2, method: 'method2', args: {} },
];

// Execute in sequence
for (const op of operations) {
    await runtime.callContract(op.address, op.method, op.args);
}
```

## Browser Compatibility

- Chrome 57+
- Firefox 52+
- Safari 11+
- Edge 16+

Requires WebAssembly support.

## Node.js Compatibility

- Node.js 12+
- Requires `--experimental-wasm-modules` flag for Node.js < 16

## Development

### Building

```bash
# Development build
wasm-pack build --dev --target web

# Production build
wasm-pack build --release --target web

# With debug symbols
wasm-pack build --debug --target web
```

### Testing

```bash
# Run Rust tests
cargo test

# Run WASM tests in browser
wasm-pack test --headless --firefox

# Run Node.js tests
node tests/node_test.js
```

### Debugging

```javascript
// Enable debug mode
const config = new AugustiumConfig();
config.debugMode = true;

const runtime = new AugustiumRuntime(config);

// Debug information will be logged to console
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run the test suite
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions and support:

- GitHub Issues: [Report bugs and request features](https://github.com/augustium/augustium/issues)
- Documentation: [Full documentation](https://docs.augustium.org)
- Community: [Join our Discord](https://discord.gg/augustium)

## Changelog

### v1.0.0
- Initial release
- Complete AVM WASM bindings
- JavaScript API layer
- TypeScript definitions
- Web integration examples
- State compatibility layer