// TypeScript definitions for Augustium WASM
// Generated automatically - do not edit manually

export interface CompileOptions {
  optimize?: boolean;
  debug?: boolean;
  target?: 'wasm' | 'bytecode';
  include_source_map?: boolean;
}

export interface CompileResult {
  success: boolean;
  bytecode: Uint8Array;
  wasm?: Uint8Array;
  errors: string[];
  warnings: string[];
  source_map?: string;
}

export interface ContractState {
  [key: string]: any;
}

export interface MethodArguments {
  [key: string]: any;
}

export interface ExecutionResult {
  success: boolean;
  result?: any;
  error?: string;
  gasUsed: number;
  events: Event[];
}

export interface Event {
  name: string;
  data: any;
  address: string;
  blockNumber?: number;
  transactionHash?: string;
}

export interface TransferEvent extends Event {
  name: 'transfer';
  data: {
    from: string;
    to: string;
    amount: number;
  };
}

export interface RuntimeStats {
  gasUsed: number;
  contractsDeployed: number;
  eventListeners: number;
}

export interface StateExport {
  avmState: any;
  config: AugustiumConfig;
  contracts: string[];
}

// Core WASM classes
export class AugustiumConfig {
  constructor();
  gasLimit: number;
  debugMode: boolean;
  enableEvents: boolean;
  maxStackSize: number;
  networkId: number;
}

export class WasmExecutionContext {
  constructor();
  setGasLimit(limit: number): void;
  getGasUsed(): number;
  setDebugMode(enabled: boolean): void;
  reset(): void;
}

export class WasmAvmState {
  constructor();
  serialize(): Uint8Array;
  static deserialize(data: Uint8Array): WasmAvmState;
  isCompatible(other: WasmAvmState): boolean;
  getVersion(): string;
}

export class WasmAvm {
  constructor();
  executeBytecode(bytecode: Uint8Array): Promise<any>;
  executeInstruction(opcode: number, operands: Uint8Array): Promise<any>;
  pushStack(value: any): void;
  popStack(): any;
  peekStack(): any;
  getStackSize(): number;
  clearStack(): void;
  deployContract(bytecode: Uint8Array, address: Uint8Array): Promise<void>;
  callContract(address: Uint8Array, method: string, args: any): Promise<any>;
  getBalance(address: Uint8Array): Promise<number>;
  transfer(from: Uint8Array, to: Uint8Array, amount: number): Promise<void>;
  getEvents(): any;
  clearEvents(): void;
  getState(): Promise<any>;
  setState(state: any): Promise<void>;
  setContext(context: WasmExecutionContext): void;
  setDebugMode(enabled: boolean): void;
  getGasUsed(): number;
  reset(): void;
}

// High-level JavaScript API
export class AugustiumRuntime {
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

export class ContractHelper {
  constructor(runtime: AugustiumRuntime, address: string);
  
  call(method: string, args: any): Promise<ExecutionResult>;
  readonly address: string;
  balance(): number;
}

// Utility functions
export class WebUtils {
  static generateAddress(): string;
  static validateAddress(address: string): boolean;
  static toHex(value: number): string;
  static fromHex(hexStr: string): number;
  static now(): number;
  static createTxHash(from: string, to: string, value: number, nonce: number): string;
}

export class AvmUtils {
  static createTestBytecode(): Uint8Array;
  static validateBytecode(bytecode: Uint8Array): boolean;
  static getVersionInfo(): any;
}

export class StateSerializer {
  static serialize(state: WasmAvmState): Uint8Array;
  static deserialize(data: Uint8Array): WasmAvmState;
  static validateCompatibility(state1: WasmAvmState, state2: WasmAvmState): boolean;
}

// Legacy API (for backward compatibility)
export class AugustiumContract {
  constructor(bytecode: Uint8Array);
  call(method: string, args: MethodArguments): Promise<any>;
  getState(): ContractState;
  setState(state: ContractState): void;
}

// Compiler functions
export function compileAugustiumToWasm(source: string, options: CompileOptions): Promise<CompileResult>;
export function parseAugustium(source: string): Promise<any>;
export function validateAugustium(source: string): Promise<{ valid: boolean; errors: string[] }>;
export function getCompilerInfo(): any;

// Initialization
export function initJsApi(): void;
export function getVersionInfo(): {
  version: string;
  name: string;
  description: string;
};

// Type guards
export function isTransferEvent(event: Event): event is TransferEvent;

// Constants
export const DEFAULT_GAS_LIMIT: number;
export const MAX_STACK_SIZE: number;
export const WASM_PAGE_SIZE: number;

// Error types
export class AugustiumError extends Error {
  constructor(message: string, code?: string);
  code?: string;
}

export class CompilationError extends AugustiumError {
  constructor(message: string, line?: number, column?: number);
  line?: number;
  column?: number;
}

export class RuntimeError extends AugustiumError {
  constructor(message: string, gasUsed?: number);
  gasUsed?: number;
}

export class ContractError extends AugustiumError {
  constructor(message: string, address?: string);
  address?: string;
}

// Event listener types
export type EventListener<T = any> = (data: T) => void;
export type TransferEventListener = EventListener<TransferEvent['data']>;

// Configuration interfaces
export interface NetworkConfig {
  id: number;
  name: string;
  rpcUrl?: string;
  blockExplorer?: string;
}

export interface GasConfig {
  limit: number;
  price: number;
  multiplier?: number;
}

// Advanced features
export interface DebugInfo {
  stackTrace: string[];
  gasUsage: number[];
  memoryUsage: number;
  executionTime: number;
}

export interface ProfilingResult {
  totalTime: number;
  gasEfficiency: number;
  memoryPeak: number;
  instructionCount: number;
}

// ML Integration (if ml features are enabled)
export interface MLTensor {
  shape: number[];
  data: Float32Array;
  dtype: 'float32' | 'int32' | 'bool';
}

export interface MLModel {
  predict(input: MLTensor): Promise<MLTensor>;
  train(inputs: MLTensor[], targets: MLTensor[]): Promise<void>;
  save(): Promise<Uint8Array>;
  load(data: Uint8Array): Promise<void>;
}

// Export default runtime instance
export const augustium: AugustiumRuntime;

// Module augmentation for global scope
declare global {
  interface Window {
    Augustium?: {
      Runtime: typeof AugustiumRuntime;
      Utils: typeof WebUtils;
      Contract: typeof ContractHelper;
    };
  }
}