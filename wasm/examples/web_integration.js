/**
 * Augustium WASM Web Integration Examples
 * 
 * This file demonstrates how to use the Augustium WASM bindings
 * in web applications for smart contract development and execution.
 */

// Import the WASM module (adjust path as needed)
import init, { 
    AugustiumRuntime, 
    AugustiumConfig, 
    ContractHelper, 
    WebUtils,
    getVersionInfo 
} from '../pkg/augustium_wasm.js';

/**
 * Initialize Augustium WASM and set up the runtime
 */
async function initializeAugustium() {
    try {
        // Initialize the WASM module
        await init();
        
        console.log('Augustium WASM initialized:', getVersionInfo());
        
        // Create runtime configuration
        const config = new AugustiumConfig();
        config.gasLimit = 2000000;
        config.debugMode = true;
        config.enableEvents = true;
        
        // Create the runtime instance
        const runtime = new AugustiumRuntime(config);
        
        return runtime;
    } catch (error) {
        console.error('Failed to initialize Augustium:', error);
        throw error;
    }
}

/**
 * Example 1: Deploy and interact with a simple contract
 */
async function simpleContractExample() {
    console.log('\n=== Simple Contract Example ===');
    
    const runtime = await initializeAugustium();
    
    // Generate a contract address
    const contractAddress = WebUtils.generateAddress();
    console.log('Generated contract address:', contractAddress);
    
    // Simple contract source code
    const contractSource = `
        contract SimpleStorage {
            uint256 private value;
            
            function setValue(uint256 newValue) public {
                value = newValue;
                emit ValueChanged(newValue);
            }
            
            function getValue() public view returns (uint256) {
                return value;
            }
            
            event ValueChanged(uint256 newValue);
        }
    `;
    
    try {
        // Deploy the contract
        const deployResult = await runtime.deployContractFromSource(contractSource, contractAddress);
        console.log('Contract deployed:', deployResult);
        
        // Create a contract helper for easier interaction
        const contract = new ContractHelper(runtime, contractAddress);
        
        // Set up event listener
        runtime.on('ValueChanged', (eventData) => {
            console.log('Value changed event:', eventData);
        });
        
        // Call contract methods
        const setResult = await contract.call('setValue', { newValue: 42 });
        console.log('setValue result:', setResult);
        
        const getResult = await contract.call('getValue', {});
        console.log('getValue result:', getResult);
        
        // Check contract balance
        const balance = contract.balance();
        console.log('Contract balance:', balance);
        
    } catch (error) {
        console.error('Contract interaction failed:', error);
    }
}

/**
 * Example 2: Token transfer and balance management
 */
async function tokenTransferExample() {
    console.log('\n=== Token Transfer Example ===');
    
    const runtime = await initializeAugustium();
    
    // Generate addresses
    const alice = WebUtils.generateAddress();
    const bob = WebUtils.generateAddress();
    
    console.log('Alice address:', alice);
    console.log('Bob address:', bob);
    
    // Set up transfer event listener
    runtime.on('transfer', (eventData) => {
        console.log('Transfer event:', {
            from: eventData.from,
            to: eventData.to,
            amount: eventData.amount
        });
    });
    
    try {
        // Check initial balances
        console.log('Initial balances:');
        console.log('Alice:', runtime.getBalance(alice));
        console.log('Bob:', runtime.getBalance(bob));
        
        // Perform transfer
        const transferAmount = 1000;
        runtime.transfer(alice, bob, transferAmount);
        
        // Check balances after transfer
        console.log('Balances after transfer:');
        console.log('Alice:', runtime.getBalance(alice));
        console.log('Bob:', runtime.getBalance(bob));
        
    } catch (error) {
        console.error('Transfer failed:', error);
    }
}

/**
 * Example 3: State management and persistence
 */
async function stateManagementExample() {
    console.log('\n=== State Management Example ===');
    
    const runtime = await initializeAugustium();
    
    // Deploy a contract and perform some operations
    const contractAddress = WebUtils.generateAddress();
    const simpleContract = new Uint8Array([0x01, 0x02, 0x03]); // Mock bytecode
    
    try {
        runtime.deployContract(simpleContract, contractAddress);
        
        // Perform some operations to change state
        await runtime.executeBytecode(new Uint8Array([0x60, 0x01, 0x60, 0x02, 0x01])); // Mock operations
        
        // Export current state
        const exportedState = runtime.exportState();
        console.log('Exported state:', exportedState);
        
        // Reset runtime
        runtime.reset();
        console.log('Runtime reset');
        
        // Import the state back
        runtime.importState(exportedState);
        console.log('State imported successfully');
        
        // Verify state restoration
        const stats = runtime.getStats();
        console.log('Runtime stats after import:', stats);
        
    } catch (error) {
        console.error('State management failed:', error);
    }
}

/**
 * Example 4: Advanced bytecode execution
 */
async function bytecodeExecutionExample() {
    console.log('\n=== Bytecode Execution Example ===');
    
    const runtime = await initializeAugustium();
    
    // Create test bytecode
    const testBytecode = new Uint8Array([
        0x60, 0x10, // PUSH1 0x10
        0x60, 0x20, // PUSH1 0x20
        0x01,       // ADD
        0x60, 0x00, // PUSH1 0x00
        0x52,       // MSTORE
        0x60, 0x20, // PUSH1 0x20
        0x60, 0x00, // PUSH1 0x00
        0xf3        // RETURN
    ]);
    
    try {
        console.log('Executing bytecode...');
        const result = await runtime.executeBytecode(testBytecode);
        console.log('Execution result:', result);
        
        // Check gas usage
        const stats = runtime.getStats();
        console.log('Gas used:', stats.gasUsed);
        
        // Get execution events
        const events = runtime.getEvents();
        console.log('Execution events:', events);
        
    } catch (error) {
        console.error('Bytecode execution failed:', error);
    }
}

/**
 * Example 5: Utility functions demonstration
 */
function utilityFunctionsExample() {
    console.log('\n=== Utility Functions Example ===');
    
    // Address generation and validation
    const address1 = WebUtils.generateAddress();
    const address2 = WebUtils.generateAddress();
    
    console.log('Generated addresses:');
    console.log('Address 1:', address1, 'Valid:', WebUtils.validateAddress(address1));
    console.log('Address 2:', address2, 'Valid:', WebUtils.validateAddress(address2));
    
    // Hex conversion
    const value = 12345;
    const hexValue = WebUtils.toHex(value);
    const backToValue = WebUtils.fromHex(hexValue);
    
    console.log('Hex conversion:');
    console.log('Original:', value);
    console.log('Hex:', hexValue);
    console.log('Back to number:', backToValue);
    
    // Transaction hash creation
    const txHash = WebUtils.createTxHash(address1, address2, value, 1);
    console.log('Transaction hash:', txHash);
    
    // Timestamp
    const now = WebUtils.now();
    console.log('Current timestamp:', now);
}

/**
 * Example 6: Error handling and debugging
 */
async function errorHandlingExample() {
    console.log('\n=== Error Handling Example ===');
    
    const config = new AugustiumConfig();
    config.debugMode = true; // Enable debug mode
    const runtime = new AugustiumRuntime(config);
    
    try {
        // Try to call a non-existent contract
        const invalidAddress = '0x1234567890123456789012345678901234567890';
        await runtime.callContract(invalidAddress, 'nonExistentMethod', {});
        
    } catch (error) {
        console.log('Caught expected error:', error.message);
    }
    
    try {
        // Try to transfer with insufficient balance
        const addr1 = WebUtils.generateAddress();
        const addr2 = WebUtils.generateAddress();
        runtime.transfer(addr1, addr2, 999999999);
        
    } catch (error) {
        console.log('Caught transfer error:', error.message);
    }
    
    try {
        // Try to execute invalid bytecode
        const invalidBytecode = new Uint8Array([0xFF, 0xFF, 0xFF]);
        await runtime.executeBytecode(invalidBytecode);
        
    } catch (error) {
        console.log('Caught bytecode error:', error.message);
    }
}

/**
 * Main demo function that runs all examples
 */
async function runAllExamples() {
    console.log('Starting Augustium WASM Integration Examples...');
    
    try {
        await simpleContractExample();
        await tokenTransferExample();
        await stateManagementExample();
        await bytecodeExecutionExample();
        utilityFunctionsExample();
        await errorHandlingExample();
        
        console.log('\n=== All Examples Completed Successfully ===');
        
    } catch (error) {
        console.error('Example execution failed:', error);
    }
}

/**
 * Web-specific integration example
 */
function webIntegrationExample() {
    console.log('\n=== Web Integration Example ===');
    
    // Create UI elements
    const container = document.createElement('div');
    container.innerHTML = `
        <h2>Augustium WASM Demo</h2>
        <div>
            <button id="deploy-btn">Deploy Contract</button>
            <button id="call-btn">Call Contract</button>
            <button id="transfer-btn">Transfer Tokens</button>
        </div>
        <div id="output"></div>
    `;
    
    document.body.appendChild(container);
    
    // Set up event handlers
    let runtime;
    
    document.getElementById('deploy-btn').addEventListener('click', async () => {
        try {
            if (!runtime) {
                runtime = await initializeAugustium();
            }
            
            const address = WebUtils.generateAddress();
            const bytecode = new Uint8Array([0x01, 0x02, 0x03]);
            runtime.deployContract(bytecode, address);
            
            document.getElementById('output').innerHTML += `<p>Contract deployed at: ${address}</p>`;
        } catch (error) {
            document.getElementById('output').innerHTML += `<p>Error: ${error.message}</p>`;
        }
    });
    
    document.getElementById('call-btn').addEventListener('click', async () => {
        try {
            if (!runtime) {
                runtime = await initializeAugustium();
            }
            
            const result = await runtime.executeBytecode(new Uint8Array([0x60, 0x01]));
            document.getElementById('output').innerHTML += `<p>Execution result: ${JSON.stringify(result)}</p>`;
        } catch (error) {
            document.getElementById('output').innerHTML += `<p>Error: ${error.message}</p>`;
        }
    });
    
    document.getElementById('transfer-btn').addEventListener('click', () => {
        try {
            if (!runtime) {
                document.getElementById('output').innerHTML += '<p>Please deploy a contract first</p>';
                return;
            }
            
            const from = WebUtils.generateAddress();
            const to = WebUtils.generateAddress();
            runtime.transfer(from, to, 100);
            
            document.getElementById('output').innerHTML += `<p>Transferred 100 tokens from ${from} to ${to}</p>`;
        } catch (error) {
            document.getElementById('output').innerHTML += `<p>Error: ${error.message}</p>`;
        }
    });
}

// Export functions for use in other modules
export {
    initializeAugustium,
    simpleContractExample,
    tokenTransferExample,
    stateManagementExample,
    bytecodeExecutionExample,
    utilityFunctionsExample,
    errorHandlingExample,
    runAllExamples,
    webIntegrationExample
};

// Auto-run examples if this script is loaded directly
if (typeof window !== 'undefined') {
    // Browser environment
    window.addEventListener('load', () => {
        console.log('Augustium WASM examples loaded. Call runAllExamples() to start.');
        
        // Expose functions to global scope for easy testing
        window.AugustiumExamples = {
            runAllExamples,
            simpleContractExample,
            tokenTransferExample,
            stateManagementExample,
            bytecodeExecutionExample,
            utilityFunctionsExample,
            errorHandlingExample,
            webIntegrationExample
        };
    });
} else if (typeof process !== 'undefined') {
    // Node.js environment
    runAllExamples().catch(console.error);
}