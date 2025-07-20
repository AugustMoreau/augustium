# Deployment Guide

Complete guide for deploying Augustium contracts to various blockchain networks.

## Table of Contents

- [Overview](#overview)
- [Supported Networks](#supported-networks)
- [Basic Deployment](#basic-deployment)
- [Multi-Chain Deployment](#multi-chain-deployment)
- [Production Deployment](#production-deployment)
- [Verification](#verification)
- [Monitoring](#monitoring)

## Overview

Augustium contracts compile to bytecode that can be deployed to:
- **Ethereum** and EVM-compatible chains
- **Native Augustium networks** (when available)
- **Test networks** for development

## Supported Networks

### Mainnet Networks
- **Ethereum** (Chain ID: 1)
- **Binance Smart Chain** (Chain ID: 56)  
- **Polygon** (Chain ID: 137)
- **Avalanche C-Chain** (Chain ID: 43114)
- **Arbitrum One** (Chain ID: 42161)
- **Optimism** (Chain ID: 10)

### Testnets
- **Ethereum Goerli** (Chain ID: 5)
- **Ethereum Sepolia** (Chain ID: 11155111)
- **BSC Testnet** (Chain ID: 97)
- **Polygon Mumbai** (Chain ID: 80001)
- **Avalanche Fuji** (Chain ID: 43113)

## Basic Deployment

### 1. Prepare Your Contract

```augustium
// contracts/MyToken.aug
contract MyToken {
    name: string = "My Token";
    symbol: string = "MTK";
    decimals: u8 = 18;
    total_supply: u256;
    
    constructor(initial_supply: u256) {
        total_supply = initial_supply;
        balances[msg.sender] = initial_supply;
    }
    
    // ... contract implementation
}
```

### 2. Configure Deployment

Create `deploy.toml`:

```toml
[deployment]
network = "ethereum-goerli"
gas_price = "20"  # gwei
gas_limit = "3000000"

[contracts.MyToken]
args = [1000000]  # Constructor arguments

[accounts]
deployer = "0x742d35Cc6634C0532925a3b8D57deFF1c1fF4dc3"
private_key_env = "DEPLOYER_PRIVATE_KEY"
```

### 3. Deploy

```bash
# Set your private key
export DEPLOYER_PRIVATE_KEY="your_private_key_here"

# Deploy to testnet
august deploy --config deploy.toml

# Deploy to mainnet (be careful!)
august deploy --config deploy.toml --network ethereum-mainnet
```

## Multi-Chain Deployment

### Configuration

```toml
# multi-chain.toml
[deployment]
networks = ["ethereum", "bsc", "polygon"]

[contracts.MyToken]
args = [1000000]
verify = true

[networks.ethereum]
rpc_url = "https://mainnet.infura.io/v3/YOUR_KEY"
gas_price = "30"

[networks.bsc]
rpc_url = "https://bsc-dataseed.binance.org"
gas_price = "5"

[networks.polygon]
rpc_url = "https://polygon-rpc.com"
gas_price = "1"
```

### Deploy to All Networks

```bash
august deploy --multi-chain multi-chain.toml
```

This will:
1. **Compile** your contract once
2. **Deploy** to all specified networks
3. **Verify** contracts on block explorers
4. **Save** deployment addresses and transaction hashes

## Production Deployment

### Security Checklist

Before deploying to mainnet:

- [ ] **Audit your contract** - Have it reviewed by security experts
- [ ] **Test thoroughly** - Deploy and test on testnets
- [ ] **Check gas limits** - Ensure sufficient gas for deployment
- [ ] **Verify constructor args** - Double-check all parameters
- [ ] **Use hardware wallet** - For production private keys
- [ ] **Set up monitoring** - Track contract behavior
- [ ] **Prepare for upgrades** - If using upgradeable contracts

### Production Configuration

```toml
[deployment]
network = "ethereum-mainnet"
gas_price = "auto"  # Use network-recommended gas price
gas_limit = "auto"  # Estimate gas automatically
confirmation_blocks = 12  # Wait for more confirmations

[contracts.MyToken]
args = [1000000000]  # 1B token supply
verify = true
optimizer = true
optimization_runs = 200

[security]
check_constructor_args = true
require_confirmation = true
max_gas_price = "100"  # gwei - prevent accidents

[accounts]
deployer = "hardware:ledger:0"  # Use hardware wallet
```

### Deploy with Confirmation

```bash
august deploy --config production.toml --confirm-each-step
```

## Verification

### Automatic Verification

Contracts are automatically verified on:
- **Etherscan** (Ethereum)
- **BscScan** (BSC)
- **PolygonScan** (Polygon)
- **SnowTrace** (Avalanche)

### Manual Verification

If automatic verification fails:

```bash
# Verify specific contract
august verify --contract MyToken --address 0x123... --network ethereum

# Verify with constructor arguments
august verify --contract MyToken --address 0x123... --args 1000000
```

### Verification Status

```bash
# Check verification status
august status --deployment-id dep_123456

# View all deployments
august deployments list
```

## Monitoring

### Set Up Monitoring

```toml
# monitoring.toml
[monitoring]
enabled = true
webhook_url = "https://hooks.slack.com/your-webhook"

[alerts]
failed_transactions = true
unusual_activity = true
gas_price_spikes = true

[contracts.MyToken]
track_events = ["Transfer", "Approval"]
alert_on_large_transfers = "1000000"  # Alert on transfers > 1M tokens
```

### Monitor Events

```bash
# Start monitoring
august monitor --config monitoring.toml

# View recent events
august events --contract 0x123... --last 24h
```

### Health Checks

```bash
# Check contract health
august health-check --contract 0x123...

# Generate deployment report
august report --deployment-id dep_123456
```

## Advanced Features

### Upgradeable Contracts

```augustium
// Use proxy pattern for upgrades
contract MyTokenV1 {
    // Implementation
}

contract MyTokenProxy {
    // Proxy logic
}
```

### Cross-Chain Deployment

```toml
[cross_chain]
enabled = true
bridge_contracts = true

[networks]
primary = "ethereum"
secondary = ["bsc", "polygon"]
```

### Batch Deployment

```bash
# Deploy multiple contracts
august batch-deploy contracts/*.aug --config batch.toml
```

## Troubleshooting

### Common Issues

**Out of Gas:**
```bash
# Increase gas limit
august deploy --gas-limit 5000000
```

**Wrong Network:**
```bash
# Verify network configuration
august network info --network ethereum
```

**Private Key Issues:**
```bash
# Test connection without deploying
august test-connection --network ethereum-goerli
```

### Gas Estimation

```bash
# Estimate deployment cost
august estimate-gas --contract MyToken

# Check current gas prices
august gas-price --network ethereum
```

### Rollback Support

```bash
# Rollback if deployment fails
august rollback --deployment-id dep_123456

# Create deployment checkpoint
august checkpoint --name "before-v2-deploy"
```

## Best Practices

1. **Always test on testnets first**
2. **Use deterministic builds** for reproducible deployments
3. **Keep deployment scripts** under version control
4. **Monitor deployed contracts** for unusual activity
5. **Have upgrade plans** ready if needed
6. **Document deployment procedures** for your team
7. **Use multi-signature wallets** for important contracts

For more advanced deployment scenarios, see the [Deployment Tools Documentation](DEPLOYMENT_TOOLS.md).
