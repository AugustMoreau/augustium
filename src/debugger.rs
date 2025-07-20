// Debugger for stepping through Augustium programs
// Set breakpoints, inspect variables, view call stack

use std::collections::{HashMap, HashSet};
use std::fmt;

use crate::avm::{AVM, Instruction, Value};
use crate::error::{CompilerError, Result};

/// Debugger for Augustium Virtual Machine
pub struct Debugger {
    vm: AVM,
    breakpoints: HashSet<usize>,
    watch_variables: HashMap<String, Value>,
    call_stack: Vec<StackFrame>,
    step_mode: StepMode,
    execution_history: Vec<ExecutionStep>,
    max_history: usize,
}

/// Stack frame information
#[derive(Debug, Clone)]
pub struct StackFrame {
    pub function_name: String,
    pub instruction_pointer: usize,
    pub local_variables: HashMap<String, Value>,
    pub parameters: Vec<Value>,
}

/// Execution step information
#[derive(Debug, Clone)]
pub struct ExecutionStep {
    pub instruction_pointer: usize,
    pub instruction: Instruction,
    pub stack_before: Vec<Value>,
    pub stack_after: Vec<Value>,
    pub gas_used: u64,
    pub timestamp: std::time::Instant,
}

/// Step mode for debugging
#[derive(Debug, Clone, PartialEq)]
pub enum StepMode {
    /// Continue execution until breakpoint or end
    Continue,
    /// Step to next instruction
    StepInto,
    /// Step over function calls
    StepOver,
    /// Step out of current function
    StepOut,
    /// Run until return from current function
    RunToReturn,
}

/// Debugger command
#[derive(Debug, Clone)]
pub enum DebugCommand {
    /// Set breakpoint at instruction
    SetBreakpoint(usize),
    /// Remove breakpoint
    RemoveBreakpoint(usize),
    /// List all breakpoints
    ListBreakpoints,
    /// Continue execution
    Continue,
    /// Step into next instruction
    StepInto,
    /// Step over function call
    StepOver,
    /// Step out of current function
    StepOut,
    /// Print current stack
    PrintStack,
    /// Print local variables
    PrintLocals,
    /// Print call stack
    PrintCallStack,
    /// Watch variable
    WatchVariable(String),
    /// Unwatch variable
    UnwatchVariable(String),
    /// Print watched variables
    PrintWatched,
    /// Print execution history
    PrintHistory(Option<usize>),
    /// Evaluate expression
    Evaluate(String),
    /// Print help
    Help,
    /// Quit debugger
    Quit,
}

/// Debugger result
#[derive(Debug)]
pub enum DebugResult {
    /// Continue debugging
    Continue,
    /// Execution completed
    Completed(Value),
    /// Execution stopped at breakpoint
    Breakpoint(usize),
    /// Error occurred
    Error(String),
    /// Quit requested
    Quit,
}

impl Debugger {
    /// Create a new debugger
    pub fn new() -> Self {
        Self {
            vm: AVM::new(),
            breakpoints: HashSet::new(),
            watch_variables: HashMap::new(),
            call_stack: Vec::new(),
            step_mode: StepMode::Continue,
            execution_history: Vec::new(),
            max_history: 1000,
        }
    }

    /// Start debugging a program
    pub fn debug(&mut self, bytecode: &[u8]) -> Result<()> {
        println!("🐛 Augustium Debugger Started");
        println!("Type 'help' for available commands");
        println!("Program loaded with {} bytes of bytecode", bytecode.len());
        
        // Load bytecode into VM
        // Note: This assumes AVM has a method to load bytecode
        // You may need to adjust based on actual AVM implementation
        
        self.debug_loop()
    }

    /// Main debug loop
    fn debug_loop(&mut self) -> Result<()> {
        use std::io::{self, Write};
        
        loop {
            print!("(augustium-db) ");
            io::stdout().flush().unwrap();
            
            let mut input = String::new();
            if io::stdin().read_line(&mut input).is_err() {
                break;
            }
            
            let command = self.parse_command(input.trim());
            match self.execute_command(command) {
                DebugResult::Continue => continue,
                DebugResult::Completed(value) => {
                    println!("✓ Program completed with result: {:?}", value);
                    break;
                }
                DebugResult::Breakpoint(ip) => {
                    println!("🛑 Breakpoint hit at instruction {}", ip);
                    self.print_current_state();
                }
                DebugResult::Error(msg) => {
                    println!("❌ Error: {}", msg);
                }
                DebugResult::Quit => {
                    println!("👋 Debugger exiting");
                    break;
                }
            }
        }
        
        Ok(())
    }

    /// Parse debug command from input
    fn parse_command(&self, input: &str) -> DebugCommand {
        let parts: Vec<&str> = input.split_whitespace().collect();
        if parts.is_empty() {
            return DebugCommand::Help;
        }

        match parts[0] {
            "b" | "break" | "breakpoint" => {
                if parts.len() > 1 {
                    if let Ok(addr) = parts[1].parse::<usize>() {
                        DebugCommand::SetBreakpoint(addr)
                    } else {
                        DebugCommand::Help
                    }
                } else {
                    DebugCommand::ListBreakpoints
                }
            }
            "rb" | "remove" => {
                if parts.len() > 1 {
                    if let Ok(addr) = parts[1].parse::<usize>() {
                        DebugCommand::RemoveBreakpoint(addr)
                    } else {
                        DebugCommand::Help
                    }
                } else {
                    DebugCommand::Help
                }
            }
            "c" | "continue" => DebugCommand::Continue,
            "s" | "step" | "stepi" => DebugCommand::StepInto,
            "n" | "next" => DebugCommand::StepOver,
            "finish" | "stepout" => DebugCommand::StepOut,
            "stack" | "bt" | "backtrace" => DebugCommand::PrintCallStack,
            "locals" | "info locals" => DebugCommand::PrintLocals,
            "print" | "p" => {
                if parts.len() > 1 {
                    DebugCommand::Evaluate(parts[1..].join(" "))
                } else {
                    DebugCommand::PrintStack
                }
            }
            "watch" | "w" => {
                if parts.len() > 1 {
                    DebugCommand::WatchVariable(parts[1].to_string())
                } else {
                    DebugCommand::PrintWatched
                }
            }
            "unwatch" | "uw" => {
                if parts.len() > 1 {
                    DebugCommand::UnwatchVariable(parts[1].to_string())
                } else {
                    DebugCommand::Help
                }
            }
            "history" | "h" => {
                let count = if parts.len() > 1 {
                    parts[1].parse().ok()
                } else {
                    None
                };
                DebugCommand::PrintHistory(count)
            }
            "help" | "?" => DebugCommand::Help,
            "quit" | "q" | "exit" => DebugCommand::Quit,
            _ => DebugCommand::Help,
        }
    }

    /// Execute debug command
    fn execute_command(&mut self, command: DebugCommand) -> DebugResult {
        match command {
            DebugCommand::SetBreakpoint(addr) => {
                self.breakpoints.insert(addr);
                println!("✓ Breakpoint set at instruction {}", addr);
                DebugResult::Continue
            }
            DebugCommand::RemoveBreakpoint(addr) => {
                if self.breakpoints.remove(&addr) {
                    println!("✓ Breakpoint removed from instruction {}", addr);
                } else {
                    println!("⚠️  No breakpoint at instruction {}", addr);
                }
                DebugResult::Continue
            }
            DebugCommand::ListBreakpoints => {
                if self.breakpoints.is_empty() {
                    println!("No breakpoints set");
                } else {
                    println!("Breakpoints:");
                    for &addr in &self.breakpoints {
                        println!("  {}", addr);
                    }
                }
                DebugResult::Continue
            }
            DebugCommand::Continue => {
                self.step_mode = StepMode::Continue;
                self.execute_until_break()
            }
            DebugCommand::StepInto => {
                self.step_mode = StepMode::StepInto;
                self.execute_single_step()
            }
            DebugCommand::StepOver => {
                self.step_mode = StepMode::StepOver;
                self.execute_single_step()
            }
            DebugCommand::StepOut => {
                self.step_mode = StepMode::StepOut;
                self.execute_until_break()
            }
            DebugCommand::PrintStack => {
                self.print_stack();
                DebugResult::Continue
            }
            DebugCommand::PrintLocals => {
                self.print_locals();
                DebugResult::Continue
            }
            DebugCommand::PrintCallStack => {
                self.print_call_stack();
                DebugResult::Continue
            }
            DebugCommand::WatchVariable(name) => {
                // This would need to be implemented based on VM state
                println!("✓ Watching variable: {}", name);
                DebugResult::Continue
            }
            DebugCommand::UnwatchVariable(name) => {
                self.watch_variables.remove(&name);
                println!("✓ Stopped watching variable: {}", name);
                DebugResult::Continue
            }
            DebugCommand::PrintWatched => {
                self.print_watched_variables();
                DebugResult::Continue
            }
            DebugCommand::PrintHistory(count) => {
                self.print_execution_history(count);
                DebugResult::Continue
            }
            DebugCommand::Evaluate(expr) => {
                println!("Expression evaluation: {} (not yet implemented)", expr);
                DebugResult::Continue
            }
            DebugCommand::Help => {
                self.print_help();
                DebugResult::Continue
            }
            DebugCommand::Quit => DebugResult::Quit,
        }
    }

    /// Execute until breakpoint or completion
    fn execute_until_break(&mut self) -> DebugResult {
        // This would integrate with the actual VM execution
        // For now, return a placeholder
        println!("Executing until breakpoint...");
        DebugResult::Continue
    }

    /// Execute a single step
    fn execute_single_step(&mut self) -> DebugResult {
        // This would execute one instruction in the VM
        println!("Stepping to next instruction...");
        self.print_current_state();
        DebugResult::Continue
    }

    /// Print current debugger state
    fn print_current_state(&self) {
        println!("📍 Current State:");
        println!("   Instruction Pointer: {}", self.get_current_ip());
        println!("   Step Mode: {:?}", self.step_mode);
        if !self.call_stack.is_empty() {
            println!("   Current Function: {}", self.call_stack.last().unwrap().function_name);
        }
    }

    /// Get current instruction pointer
    fn get_current_ip(&self) -> usize {
        // This would get the IP from the VM
        0
    }

    /// Print VM stack
    fn print_stack(&self) {
        println!("📚 VM Stack:");
        // This would print the actual VM stack
        println!("   (Stack contents would be shown here)");
    }

    /// Print local variables
    fn print_locals(&self) {
        println!("🔍 Local Variables:");
        if let Some(frame) = self.call_stack.last() {
            for (name, value) in &frame.local_variables {
                println!("   {} = {:?}", name, value);
            }
        } else {
            println!("   No local variables (not in function)");
        }
    }

    /// Print call stack
    fn print_call_stack(&self) {
        println!("📞 Call Stack:");
        if self.call_stack.is_empty() {
            println!("   (empty)");
        } else {
            for (i, frame) in self.call_stack.iter().enumerate() {
                println!("   #{}: {} at IP {}", i, frame.function_name, frame.instruction_pointer);
            }
        }
    }

    /// Print watched variables
    fn print_watched_variables(&self) {
        println!("👁️  Watched Variables:");
        if self.watch_variables.is_empty() {
            println!("   (none)");
        } else {
            for (name, value) in &self.watch_variables {
                println!("   {} = {:?}", name, value);
            }
        }
    }

    /// Print execution history
    fn print_execution_history(&self, count: Option<usize>) {
        let count = count.unwrap_or(10).min(self.execution_history.len());
        println!("📜 Execution History (last {} steps):", count);
        
        for step in self.execution_history.iter().rev().take(count) {
            println!("   IP {}: {:?} (gas: {})", 
                step.instruction_pointer, 
                step.instruction, 
                step.gas_used
            );
        }
    }

    /// Print help information
    fn print_help(&self) {
        println!("🆘 Augustium Debugger Commands:");
        println!();
        println!("Breakpoints:");
        println!("  b, break <addr>     Set breakpoint at instruction address");
        println!("  rb, remove <addr>   Remove breakpoint");
        println!("  b                   List all breakpoints");
        println!();
        println!("Execution:");
        println!("  c, continue         Continue execution");
        println!("  s, step, stepi      Step into next instruction");
        println!("  n, next             Step over (don't enter functions)");
        println!("  finish, stepout     Step out of current function");
        println!();
        println!("Information:");
        println!("  stack, bt           Show call stack");
        println!("  locals              Show local variables");
        println!("  print, p [expr]     Print stack or evaluate expression");
        println!("  watch, w <var>      Watch variable");
        println!("  unwatch, uw <var>   Stop watching variable");
        println!("  history, h [n]      Show execution history");
        println!();
        println!("General:");
        println!("  help, ?             Show this help");
        println!("  quit, q, exit       Exit debugger");
    }
}

impl fmt::Display for StepMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StepMode::Continue => write!(f, "Continue"),
            StepMode::StepInto => write!(f, "Step Into"),
            StepMode::StepOver => write!(f, "Step Over"),
            StepMode::StepOut => write!(f, "Step Out"),
            StepMode::RunToReturn => write!(f, "Run to Return"),
        }
    }
}

/// Start interactive debugger
pub fn start_debugger(bytecode: &[u8]) -> Result<()> {
    let mut debugger = Debugger::new();
    debugger.debug(bytecode)
}