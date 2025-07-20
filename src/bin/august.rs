// August CLI - our custom project manager
// Like cargo but for Augustium projects

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{self, Command};

use augustc::package_manager::{create_package_manager, init_new_package, DependencySpec};
use augustc::error::{CompilerError, Result};

// Help text for august command
fn print_help() {
    println!("August v0.1.0");
    println!("Project management tool for Augustium");
    println!();
    println!("USAGE:");
    println!("    august <command> [options]");
    println!();
    println!("COMMANDS:");
    println!("    new <name>         Create new Augustium project");
    println!("    init               Initialize Augustium project in current directory");
    println!("    build              Build the current project");
    println!("    run                Build and run the main contract");
    println!("    test               Run project tests");
    println!("    clean              Clean build artifacts");
    println!("    check              Check project for errors without building");
    println!("    install <package>  Add dependency to project");
    println!("    update             Update dependencies to latest versions");
    println!("    search <query>     Search for packages in registry");
    println!("    publish            Publish package to registry");
    println!("    login              Authenticate with package registry");
    println!("    fmt                Format all source files");
    println!("    doc                Generate and open documentation");
    println!("    tree               Show dependency tree");
    println!("    version            Show version information");
    println!();
    println!("BUILD OPTIONS:");
    println!("    --release          Build with optimizations");
    println!("    --debug            Build with debug information");
    println!("    --target <target>  Build for specific target (avm, evm)");
    println!("    --features <list>  Enable specific features");
    println!("    --verbose, -v      Verbose output");
    println!("    --quiet, -q        Suppress output");
    println!();
    println!("EXAMPLES:");
    println!("    august new my_project         # Create new project");
    println!("    august build --release        # Build optimized version");
    println!("    august run                    # Build and execute main contract");
    println!("    august test                   # Run all tests");
    println!("    august install defi-utils     # Add dependency");
    println!("    august publish --dry-run      # Test package publishing");
}

/// Main entry point
fn main() {
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 {
        print_help();
        return;
    }
    
    let command = &args[1];
    
    match command.as_str() {
        "help" | "--help" | "-h" => print_help(),
        "version" | "--version" | "-V" => print_version(),
        "new" => handle_new(&args[2..]),
        "init" => handle_init(&args[2..]),
        "build" => handle_build(&args[2..]),
        "run" => handle_run(&args[2..]),
        "test" => handle_test(&args[2..]),
        "clean" => handle_clean(&args[2..]),
        "check" => handle_check(&args[2..]),
        "install" => handle_install(&args[2..]),
        "update" => handle_update(&args[2..]),
        "search" => handle_search(&args[2..]),
        "publish" => handle_publish(&args[2..]),
        "login" => handle_login(&args[2..]),
        "fmt" => handle_format(&args[2..]),
        "doc" => handle_doc(&args[2..]),
        "tree" => handle_tree(&args[2..]),
        _ => {
            eprintln!("❌ Unknown command: {}", command);
            eprintln!("Run 'august help' for usage information.");
            process::exit(1);
        }
    }
}

/// Print version information
fn print_version() {
    println!("august 0.1.0");
    println!("augustc 0.1.0");
    println!("Augustium toolchain 2024");
}

/// Handle new project creation
fn handle_new(args: &[String]) {
    if args.is_empty() {
        eprintln!("❌ Project name required");
        eprintln!("Usage: august new <name>");
        process::exit(1);
    }
    
    let project_name = &args[0];
    let project_path = PathBuf::from(format!("./{}", project_name));
    
    if project_path.exists() {
        eprintln!("❌ Directory '{}' already exists", project_name);
        process::exit(1);
    }
    
    // Create the project directory first
    if let Err(e) = fs::create_dir_all(&project_path) {
        eprintln!("❌ Failed to create directory: {}", e);
        process::exit(1);
    }
    
    // Create standard directory structure
    if let Err(e) = fs::create_dir_all(project_path.join("src")) {
        eprintln!("❌ Failed to create src directory: {}", e);
        process::exit(1);
    }
    
    if let Err(e) = fs::create_dir_all(project_path.join("tests")) {
        eprintln!("❌ Failed to create tests directory: {}", e);
        process::exit(1);
    }
    
    // Create main.aug file
    let main_content = format!(
        "// {}
// Main contract file

contract {} {{
    
    // Main function
    pub fn main() {{
        // Your code here - basic arithmetic
        let x = 42;
        let y = x + 8;
        // Add your contract logic here
    }}
    
    // Add your functions here
    pub fn add(a: u32, b: u32) -> u32 {{
        return a + b;
    }}
    
    pub fn multiply(a: u32, b: u32) -> u32 {{
        return a * b;
    }}
}}",
        project_name, project_name
    );
    
    if let Err(e) = fs::write(project_path.join("src").join("main.aug"), main_content) {
        eprintln!("❌ Failed to create main.aug: {}", e);
        process::exit(1);
    }
    
    match init_new_package(project_name, &project_path) {
        Ok(()) => {
            println!("✅ Created new Augustium project: {}", project_name);
            println!("   📁 Project directory: {}", project_path.display());
            println!();
            println!("Next steps:");
            println!("   cd {}", project_name);
            println!("   august build");
        }
        Err(e) => {
            eprintln!("❌ Failed to create project: {}", e);
            process::exit(1);
        }
    }
}

/// Handle project initialization
fn handle_init(args: &[String]) {
    let verbose = args.contains(&"--verbose".to_string()) || args.contains(&"-v".to_string());
    
    let current_dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let project_name = current_dir.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("augustium-project");
    
    if verbose {
        println!("🔧 Initializing Augustium project in current directory...");
    }
    
    match init_new_package(project_name, &current_dir) {
        Ok(()) => {
            println!("✅ Initialized Augustium project: {}", project_name);
        }
        Err(e) => {
            eprintln!("❌ Failed to initialize project: {}", e);
            process::exit(1);
        }
    }
}

/// Handle project build
fn handle_build(args: &[String]) {
    let release = args.contains(&"--release".to_string());
    let debug = args.contains(&"--debug".to_string());
    let verbose = args.contains(&"--verbose".to_string()) || args.contains(&"-v".to_string());
    let quiet = args.contains(&"--quiet".to_string()) || args.contains(&"-q".to_string());
    
    let target = get_flag_value(args, "--target").unwrap_or("avm".to_string());
    let features = get_flag_value(args, "--features");
    
    if !quiet {
        if release {
            println!("🚀 Building Augustium project (release mode)...");
        } else {
            println!("🔨 Building Augustium project (debug mode)...");
        }
    }
    
    // Find Aug.toml to determine project structure
    let aug_toml = find_project_config();
    if aug_toml.is_none() {
        eprintln!("❌ No Aug.toml found. Run 'august init' to create a project.");
        process::exit(1);
    }
    
    // Build the project using augustc
    let mut cmd_args = vec!["build".to_string()];
    
    if release {
        cmd_args.push("--optimize".to_string());
    }
    if debug {
        cmd_args.push("--debug".to_string());
    }
    if verbose {
        cmd_args.push("--verbose".to_string());
    }
    if quiet {
        cmd_args.push("--quiet".to_string());
    }
    
    cmd_args.push("--target".to_string());
    cmd_args.push(target);
    
    if let Some(features) = features {
        cmd_args.push("--features".to_string());
        cmd_args.push(features);
    }
    
    match run_augustc(&cmd_args) {
        Ok(()) => {
            if !quiet {
                println!("✅ Build completed successfully!");
            }
        }
        Err(e) => {
            eprintln!("❌ Build failed: {}", e);
            process::exit(1);
        }
    }
}

/// Handle run command
fn handle_run(args: &[String]) {
    let release = args.contains(&"--release".to_string());
    let verbose = args.contains(&"--verbose".to_string()) || args.contains(&"-v".to_string());
    
    if verbose {
        println!("🏃 Building and running Augustium project...");
    }
    
    // First build the project
    let mut build_args = vec!["--quiet".to_string()];
    if release {
        build_args.push("--release".to_string());
    }
    handle_build(&build_args);
    
    // Find the main contract file
    let main_contract = find_main_contract();
    if main_contract.is_none() {
        eprintln!("❌ No main contract found. Expected src/main.aug or src/lib.aug");
        process::exit(1);
    }
    
    let main_file = main_contract.unwrap();
    let mut cmd_args = vec!["run".to_string(), main_file];
    
    if verbose {
        cmd_args.push("--debug".to_string());
    }
    
    match run_augustc(&cmd_args) {
        Ok(()) => {
            if verbose {
                println!("✅ Execution completed!");
            }
        }
        Err(e) => {
            eprintln!("❌ Execution failed: {}", e);
            process::exit(1);
        }
    }
}

/// Handle test command
fn handle_test(args: &[String]) {
    let verbose = args.contains(&"--verbose".to_string()) || args.contains(&"-v".to_string());
    let release = args.contains(&"--release".to_string());
    
    if verbose {
        println!("🧪 Running Augustium tests...");
    }
    
    let mut cmd_args = vec!["test".to_string()];
    
    if verbose {
        cmd_args.push("--verbose".to_string());
    }
    if release {
        cmd_args.push("--optimize".to_string());
    }
    
    match run_augustc(&cmd_args) {
        Ok(()) => {
            println!("✅ All tests passed!");
        }
        Err(e) => {
            eprintln!("❌ Tests failed: {}", e);
            process::exit(1);
        }
    }
}

/// Handle clean command
fn handle_clean(args: &[String]) {
    let verbose = args.contains(&"--verbose".to_string()) || args.contains(&"-v".to_string());
    
    if verbose {
        println!("🧹 Cleaning build artifacts...");
    }
    
    // Remove common build directories
    let dirs_to_clean = vec!["target", "build", "dist", "*.avm"];
    
    for dir in dirs_to_clean {
        if dir.contains('*') {
            // Handle glob patterns for files
            if let Ok(entries) = fs::read_dir(".") {
                for entry in entries {
                    if let Ok(entry) = entry {
                        let path = entry.path();
                        if let Some(ext) = path.extension() {
                            if dir.ends_with(&format!("*.{}", ext.to_string_lossy())) {
                                let _ = fs::remove_file(&path);
                                if verbose {
                                    println!("   Removed: {}", path.display());
                                }
                            }
                        }
                    }
                }
            }
        } else {
            if Path::new(dir).exists() {
                let _ = fs::remove_dir_all(dir);
                if verbose {
                    println!("   Removed: {}/", dir);
                }
            }
        }
    }
    
    match run_augustc(&["clean".to_string()]) {
        Ok(()) => println!("✅ Clean completed!"),
        Err(e) => {
            if verbose {
                eprintln!("⚠️  augustc clean failed: {}", e);
            }
        }
    }
}

/// Handle check command
fn handle_check(args: &[String]) {
    let verbose = args.contains(&"--verbose".to_string()) || args.contains(&"-v".to_string());
    
    if verbose {
        println!("🔍 Checking Augustium project...");
    }
    
    // Use compile --check flag
    let main_contract = find_main_contract().unwrap_or("src/main.aug".to_string());
    let mut cmd_args = vec!["compile".to_string(), main_contract, "--check".to_string()];
    
    if verbose {
        cmd_args.push("--debug".to_string());
    } else {
        cmd_args.push("--quiet".to_string());
    }
    
    match run_augustc(&cmd_args) {
        Ok(()) => {
            if verbose {
                println!("✅ Check completed - no errors found!");
            }
        }
        Err(e) => {
            eprintln!("❌ Check failed: {}", e);
            process::exit(1);
        }
    }
}

/// Handle dependency installation
fn handle_install(args: &[String]) {
    if args.is_empty() {
        eprintln!("❌ Package name required");
        eprintln!("Usage: august install <package>");
        process::exit(1);
    }
    
    let package_name = &args[0];
    let version = get_flag_value(args, "--version");
    let verbose = args.contains(&"--verbose".to_string()) || args.contains(&"-v".to_string());
    
    if verbose {
        println!("📦 Installing package: {}", package_name);
    }
    
    let mut pm = create_package_manager();
    
    let spec = if let Some(version) = version {
        DependencySpec::Simple(version)
    } else {
        DependencySpec::Simple("*".to_string())
    };
    
    match pm.add_dependency(package_name, spec, false) {
        Ok(_) => println!("✅ Successfully installed {}", package_name),
        Err(e) => {
            eprintln!("❌ Failed to install {}: {}", package_name, e);
            process::exit(1);
        }
    }
}

/// Handle dependency update
fn handle_update(args: &[String]) {
    let verbose = args.contains(&"--verbose".to_string()) || args.contains(&"-v".to_string());
    
    if verbose {
        println!("🔄 Updating dependencies...");
    }
    
    let mut pm = create_package_manager();
    
    match pm.update_dependencies() {
        Ok(_) => println!("✅ Dependencies updated successfully"),
        Err(e) => {
            eprintln!("❌ Failed to update dependencies: {}", e);
            process::exit(1);
        }
    }
}

/// Handle package search
fn handle_search(args: &[String]) {
    if args.is_empty() {
        eprintln!("❌ Search query required");
        eprintln!("Usage: august search <query>");
        process::exit(1);
    }
    
    let query = &args[0];
    let pm = create_package_manager();
    
    match pm.search_packages(query) {
        Ok(packages) => {
            if packages.is_empty() {
                println!("No packages found matching '{}'", query);
            } else {
                println!("Found {} packages:", packages.len());
                for pkg in packages {
                    println!("  📦 {} v{}", pkg.name, pkg.version);
                    if let Some(desc) = pkg.description {
                        println!("      {}", desc);
                    }
                }
            }
        }
        Err(e) => {
            eprintln!("❌ Search failed: {}", e);
            process::exit(1);
        }
    }
}

/// Handle package publishing
fn handle_publish(args: &[String]) {
    let dry_run = args.contains(&"--dry-run".to_string());
    let pm = create_package_manager();
    
    match pm.publish_package(dry_run) {
        Ok(()) => {
            if dry_run {
                println!("✅ Dry run completed successfully");
            } else {
                println!("✅ Package published successfully");
            }
        }
        Err(e) => {
            eprintln!("❌ Publish failed: {}", e);
            process::exit(1);
        }
    }
}

/// Handle registry login
fn handle_login(_args: &[String]) {
    println!("🔑 Registry authentication");
    println!("This feature is not yet implemented.");
    println!("Contact the Augustium team for registry access.");
}

/// Handle code formatting
fn handle_format(args: &[String]) {
    let verbose = args.contains(&"--verbose".to_string()) || args.contains(&"-v".to_string());
    
    if verbose {
        println!("✨ Formatting Augustium source files...");
    }
    
    match run_augustc(&["fmt".to_string()]) {
        Ok(()) => {
            if verbose {
                println!("✅ Formatting completed!");
            }
        }
        Err(e) => {
            eprintln!("❌ Formatting failed: {}", e);
            process::exit(1);
        }
    }
}

/// Handle documentation generation
fn handle_doc(args: &[String]) {
    let open = args.contains(&"--open".to_string());
    let verbose = args.contains(&"--verbose".to_string()) || args.contains(&"-v".to_string());
    
    if verbose {
        println!("📚 Generating documentation...");
    }
    
    // This would integrate with a doc generator
    println!("📝 Documentation generation is not yet implemented.");
    println!("Use 'augustc --help' for compiler documentation.");
    
    if open {
        println!("Would open documentation in browser...");
    }
}

/// Handle dependency tree display
fn handle_tree(_args: &[String]) {
    println!("📊 Dependency tree:");
    println!("This feature is not yet implemented.");
    println!("Use package manager commands to view dependencies.");
}

/// Helper functions

/// Run augustc with given arguments
fn run_augustc(args: &[String]) -> Result<()> {
    let augustc_path = find_augustc_binary()?;
    
    let mut cmd = Command::new(augustc_path);
    cmd.args(args);
    
    let status = cmd.status()
        .map_err(|e| CompilerError::IoError(format!("Failed to run augustc: {}", e)))?;
    
    if status.success() {
        Ok(())
    } else {
        Err(CompilerError::IoError("augustc command failed".to_string()))
    }
}

/// Find augustc binary
fn find_augustc_binary() -> Result<PathBuf> {
    // Try different locations
    let mut candidates = vec![
        PathBuf::from("./target/debug/augustc"),
        PathBuf::from("./target/release/augustc"),
    ];
    
    // Look for augustc in parent directories (for projects)
    let mut current = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    for _ in 0..5 { // Check up to 5 parent levels
        candidates.push(current.join("target/debug/augustc"));
        candidates.push(current.join("target/release/augustc"));
        if !current.pop() {
            break;
        }
    }
    
    // Try in PATH
    candidates.push(PathBuf::from("augustc"));
    
    for candidate in candidates {
        if candidate.exists() {
            return Ok(candidate);
        }
    }
    
    // Try using which command
    if let Ok(output) = Command::new("which").arg("augustc").output() {
        if output.status.success() {
            let path_output = String::from_utf8_lossy(&output.stdout);
            let path_str = path_output.trim();
            if !path_str.is_empty() {
                return Ok(PathBuf::from(path_str));
            }
        }
    }
    
    Err(CompilerError::IoError("augustc binary not found. Make sure it's built or in PATH.".to_string()))
}

/// Find project configuration file
fn find_project_config() -> Option<PathBuf> {
    let mut current = std::env::current_dir().ok()?;
    
    loop {
        let aug_toml = current.join("Aug.toml");
        if aug_toml.exists() {
            return Some(aug_toml);
        }
        
        if !current.pop() {
            break;
        }
    }
    
    None
}

/// Find main contract file
fn find_main_contract() -> Option<String> {
    let candidates = vec!["src/main.aug", "src/lib.aug", "main.aug"];
    
    for candidate in candidates {
        if Path::new(candidate).exists() {
            return Some(candidate.to_string());
        }
    }
    
    None
}

/// Get flag value from arguments
fn get_flag_value(args: &[String], flag: &str) -> Option<String> {
    for i in 0..args.len() {
        if args[i] == flag && i + 1 < args.len() {
            return Some(args[i + 1].clone());
        }
    }
    None
}
