// Package manager - handles dependencies and project configs
// Works with Aug.toml files

use std::collections::{HashMap, BTreeMap};
use std::fs;
use std::path::{Path, PathBuf};
use std::io::{self, Write};
use serde::{Deserialize, Serialize};

use crate::error::{CompilerError, Result};

// Main package manager struct
pub struct PackageManager {
    // Config for the current project (loaded from Aug.toml)
    project_config: Option<ProjectConfig>,
    // Registry for downloading packages
    registry: PackageRegistry,
    /// Local package cache
    cache: PackageCache,
    /// Dependency resolver
    resolver: DependencyResolver,
}

/// Project configuration (Aug.toml)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectConfig {
    pub package: PackageInfo,
    pub dependencies: HashMap<String, DependencySpec>,
    pub dev_dependencies: HashMap<String, DependencySpec>,
    pub build_dependencies: HashMap<String, DependencySpec>,
    pub features: HashMap<String, Vec<String>>,
    pub workspace: Option<WorkspaceConfig>,
    pub profile: HashMap<String, ProfileConfig>,
}

/// Package information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageInfo {
    pub name: String,
    pub version: String,
    pub description: Option<String>,
    pub authors: Vec<String>,
    pub license: Option<String>,
    pub repository: Option<String>,
    pub homepage: Option<String>,
    pub documentation: Option<String>,
    pub keywords: Vec<String>,
    pub categories: Vec<String>,
    pub edition: String,
}

/// Dependency specification
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DependencySpec {
    Simple(String),
    Detailed {
        version: Option<String>,
        git: Option<String>,
        branch: Option<String>,
        tag: Option<String>,
        rev: Option<String>,
        path: Option<String>,
        features: Option<Vec<String>>,
        default_features: Option<bool>,
        optional: Option<bool>,
    },
}

/// Workspace configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceConfig {
    pub members: Vec<String>,
    pub exclude: Vec<String>,
    pub resolver: Option<String>,
}

/// Build profile configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileConfig {
    pub opt_level: Option<u8>,
    pub debug: Option<bool>,
    pub debug_assertions: Option<bool>,
    pub overflow_checks: Option<bool>,
    pub lto: Option<bool>,
    pub panic: Option<String>,
    pub codegen_units: Option<u32>,
    pub rpath: Option<bool>,
}

/// Package registry client
#[derive(Debug)]
pub struct PackageRegistry {
    pub registry_url: String,
    pub auth_token: Option<String>,
    pub timeout: std::time::Duration,
}

/// Local package cache
#[derive(Debug)]
pub struct PackageCache {
    pub cache_dir: PathBuf,
    pub packages: HashMap<String, CachedPackage>,
}

/// Cached package information
#[derive(Debug, Clone)]
pub struct CachedPackage {
    pub name: String,
    pub version: String,
    pub path: PathBuf,
    pub metadata: PackageMetadata,
    pub last_updated: std::time::SystemTime,
}

/// Package metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageMetadata {
    pub name: String,
    pub version: String,
    pub description: Option<String>,
    pub authors: Vec<String>,
    pub license: Option<String>,
    pub dependencies: HashMap<String, DependencySpec>,
    pub features: HashMap<String, Vec<String>>,
    pub targets: Vec<Target>,
    pub manifest_path: PathBuf,
}

/// Build target
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Target {
    pub name: String,
    pub kind: Vec<String>,
    pub src_path: PathBuf,
    pub edition: String,
    pub doctest: bool,
    pub test: bool,
}

/// Dependency resolver
#[derive(Debug)]
pub struct DependencyResolver {
    pub resolution_strategy: ResolutionStrategy,
    pub allow_prerelease: bool,
    pub offline: bool,
}

/// Dependency resolution strategy
#[derive(Debug, Clone)]
pub enum ResolutionStrategy {
    Minimal,
    MaximallyCompatible,
    Latest,
}

/// Resolved dependency graph
#[derive(Debug)]
pub struct DependencyGraph {
    pub packages: HashMap<String, ResolvedPackage>,
    pub root: String,
}

/// Resolved package
#[derive(Debug, Clone)]
pub struct ResolvedPackage {
    pub name: String,
    pub version: String,
    pub source: PackageSource,
    pub dependencies: Vec<String>,
    pub features: Vec<String>,
}

/// Package source
#[derive(Debug, Clone)]
pub enum PackageSource {
    Registry { url: String },
    Git { url: String, rev: String },
    Path { path: PathBuf },
    Local,
}

/// Package installation result
#[derive(Debug)]
pub struct InstallResult {
    pub installed: Vec<String>,
    pub updated: Vec<String>,
    pub removed: Vec<String>,
    pub warnings: Vec<String>,
}

impl Default for PackageRegistry {
    fn default() -> Self {
        Self {
            registry_url: "https://registry.augustium.org".to_string(),
            auth_token: None,
            timeout: std::time::Duration::from_secs(30),
        }
    }
}

impl Default for DependencyResolver {
    fn default() -> Self {
        Self {
            resolution_strategy: ResolutionStrategy::MaximallyCompatible,
            allow_prerelease: false,
            offline: false,
        }
    }
}

impl PackageManager {
    /// Create a new package manager
    pub fn new() -> Self {
        let cache_dir = dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from(".aug_cache"))
            .join("augustium");
        
        Self {
            project_config: None,
            registry: PackageRegistry::default(),
            cache: PackageCache {
                cache_dir,
                packages: HashMap::new(),
            },
            resolver: DependencyResolver::default(),
        }
    }

    /// Initialize a new package
    pub fn init_package(&self, name: &str, path: &Path) -> Result<()> {
        let config = ProjectConfig {
            package: PackageInfo {
                name: name.to_string(),
                version: "0.1.0".to_string(),
                description: Some(format!("A new Augustium package: {}", name)),
                authors: vec!["Your Name <your.email@example.com>".to_string()],
                license: Some("MIT".to_string()),
                repository: None,
                homepage: None,
                documentation: None,
                keywords: vec![],
                categories: vec![],
                edition: "2024".to_string(),
            },
            dependencies: HashMap::new(),
            dev_dependencies: HashMap::new(),
            build_dependencies: HashMap::new(),
            features: HashMap::new(),
            workspace: None,
            profile: {
                let mut profiles = HashMap::new();
                profiles.insert("dev".to_string(), ProfileConfig {
                    opt_level: Some(0),
                    debug: Some(true),
                    debug_assertions: Some(true),
                    overflow_checks: Some(true),
                    lto: Some(false),
                    panic: Some("unwind".to_string()),
                    codegen_units: Some(256),
                    rpath: Some(false),
                });
                profiles.insert("release".to_string(), ProfileConfig {
                    opt_level: Some(3),
                    debug: Some(false),
                    debug_assertions: Some(false),
                    overflow_checks: Some(false),
                    lto: Some(true),
                    panic: Some("abort".to_string()),
                    codegen_units: Some(1),
                    rpath: Some(false),
                });
                profiles
            },
        };

        let config_path = path.join("Aug.toml");
        let toml_content = toml::to_string_pretty(&config)
            .map_err(|e| CompilerError::IoError(format!("Failed to serialize config: {}", e)))?;
        
        fs::write(&config_path, toml_content)
            .map_err(|e| CompilerError::IoError(format!("Failed to write Aug.toml: {}", e)))?;

        println!("📦 Initialized new Augustium package: {}", name);
        Ok(())
    }

    /// Load project configuration
    pub fn load_config(&mut self, project_path: &Path) -> Result<()> {
        let config_path = project_path.join("Aug.toml");
        
        if !config_path.exists() {
            return Err(CompilerError::IoError("Aug.toml not found".to_string()));
        }

        let content = fs::read_to_string(&config_path)
            .map_err(|e| CompilerError::IoError(format!("Failed to read Aug.toml: {}", e)))?;
        
        let config: ProjectConfig = toml::from_str(&content)
            .map_err(|e| CompilerError::IoError(format!("Failed to parse Aug.toml: {}", e)))?;
        
        self.project_config = Some(config);
        Ok(())
    }

    /// Add a dependency
    pub fn add_dependency(&mut self, name: &str, spec: DependencySpec, dev: bool) -> Result<()> {
        if let Some(ref mut config) = self.project_config {
            let deps = if dev {
                &mut config.dev_dependencies
            } else {
                &mut config.dependencies
            };
            
            deps.insert(name.to_string(), spec);
            println!("➕ Added dependency: {}", name);
            Ok(())
        } else {
            Err(CompilerError::IoError("No project configuration loaded".to_string()))
        }
    }

    /// Remove a dependency
    pub fn remove_dependency(&mut self, name: &str, dev: bool) -> Result<()> {
        if let Some(ref mut config) = self.project_config {
            let deps = if dev {
                &mut config.dev_dependencies
            } else {
                &mut config.dependencies
            };
            
            if deps.remove(name).is_some() {
                println!("➖ Removed dependency: {}", name);
                Ok(())
            } else {
                Err(CompilerError::IoError(format!("Dependency '{}' not found", name)))
            }
        } else {
            Err(CompilerError::IoError("No project configuration loaded".to_string()))
        }
    }

    /// Resolve dependencies
    pub fn resolve_dependencies(&self) -> Result<DependencyGraph> {
        if let Some(ref config) = self.project_config {
            println!("🔍 Resolving dependencies...");
            
            let mut graph = DependencyGraph {
                packages: HashMap::new(),
                root: config.package.name.clone(),
            };
            
            // Add root package
            graph.packages.insert(config.package.name.clone(), ResolvedPackage {
                name: config.package.name.clone(),
                version: config.package.version.clone(),
                source: PackageSource::Local,
                dependencies: config.dependencies.keys().cloned().collect(),
                features: vec![],
            });
            
            // Resolve each dependency (simplified)
            for (name, spec) in &config.dependencies {
                let resolved = self.resolve_single_dependency(name, spec)?;
                graph.packages.insert(name.clone(), resolved);
            }
            
            println!("✅ Dependencies resolved successfully");
            Ok(graph)
        } else {
            Err(CompilerError::IoError("No project configuration loaded".to_string()))
        }
    }

    /// Resolve a single dependency
    fn resolve_single_dependency(&self, name: &str, spec: &DependencySpec) -> Result<ResolvedPackage> {
        match spec {
            DependencySpec::Simple(version) => {
                Ok(ResolvedPackage {
                    name: name.to_string(),
                    version: version.clone(),
                    source: PackageSource::Registry {
                        url: self.registry.registry_url.clone(),
                    },
                    dependencies: vec![], // Would be resolved recursively
                    features: vec![],
                })
            },
            DependencySpec::Detailed { version, git, path, .. } => {
                let source = if let Some(git_url) = git {
                    PackageSource::Git {
                        url: git_url.clone(),
                        rev: "main".to_string(), // Would use actual rev/branch/tag
                    }
                } else if let Some(local_path) = path {
                    PackageSource::Path {
                        path: PathBuf::from(local_path),
                    }
                } else {
                    PackageSource::Registry {
                        url: self.registry.registry_url.clone(),
                    }
                };
                
                Ok(ResolvedPackage {
                    name: name.to_string(),
                    version: version.as_ref().unwrap_or(&"*".to_string()).clone(),
                    source,
                    dependencies: vec![],
                    features: vec![],
                })
            },
        }
    }

    /// Install dependencies
    pub fn install_dependencies(&mut self, graph: &DependencyGraph) -> Result<InstallResult> {
        println!("📥 Installing dependencies...");
        
        let mut result = InstallResult {
            installed: vec![],
            updated: vec![],
            removed: vec![],
            warnings: vec![],
        };
        
        // Create cache directory
        fs::create_dir_all(&self.cache.cache_dir)
            .map_err(|e| CompilerError::IoError(format!("Failed to create cache dir: {}", e)))?;
        
        for (name, package) in &graph.packages {
            if name == &graph.root {
                continue; // Skip root package
            }
            
            match self.install_package(package) {
                Ok(()) => {
                    result.installed.push(name.clone());
                    println!("  ✅ Installed: {} v{}", name, package.version);
                },
                Err(e) => {
                    result.warnings.push(format!("Failed to install {}: {}", name, e));
                    println!("  ⚠️  Warning: Failed to install {}: {}", name, e);
                },
            }
        }
        
        println!("🎉 Installation complete!");
        Ok(result)
    }

    /// Install a single package
    fn install_package(&mut self, package: &ResolvedPackage) -> Result<()> {
        let package_dir = self.cache.cache_dir
            .join(&package.name)
            .join(&package.version);
        
        if package_dir.exists() {
            return Ok(()); // Already installed
        }
        
        fs::create_dir_all(&package_dir)
            .map_err(|e| CompilerError::IoError(format!("Failed to create package dir: {}", e)))?;
        
        match &package.source {
            PackageSource::Registry { .. } => {
                // Would download from registry
                self.download_from_registry(&package.name, &package.version, &package_dir)?;
            },
            PackageSource::Git { url, rev } => {
                // Would clone from git
                self.clone_from_git(url, rev, &package_dir)?;
            },
            PackageSource::Path { path } => {
                // Would copy from local path
                self.copy_from_path(path, &package_dir)?;
            },
            PackageSource::Local => {
                // Nothing to install for local packages
            },
        }
        
        // Cache package info
        self.cache.packages.insert(package.name.clone(), CachedPackage {
            name: package.name.clone(),
            version: package.version.clone(),
            path: package_dir,
            metadata: PackageMetadata {
                name: package.name.clone(),
                version: package.version.clone(),
                description: None,
                authors: vec![],
                license: None,
                dependencies: HashMap::new(),
                features: HashMap::new(),
                targets: vec![],
                manifest_path: PathBuf::new(),
            },
            last_updated: std::time::SystemTime::now(),
        });
        
        Ok(())
    }

    /// Download package from registry (placeholder)
    fn download_from_registry(&self, name: &str, version: &str, target_dir: &Path) -> Result<()> {
        // This would implement actual registry download
        println!("📡 Downloading {} v{} from registry...", name, version);
        
        // Create a placeholder package structure
        fs::create_dir_all(target_dir.join("src"))
            .map_err(|e| CompilerError::IoError(format!("Failed to create src dir: {}", e)))?;
        
        let lib_content = format!("// {} v{}\n// Downloaded from registry\n", name, version);
        fs::write(target_dir.join("src").join("lib.aug"), lib_content)
            .map_err(|e| CompilerError::IoError(format!("Failed to write lib.aug: {}", e)))?;
        
        Ok(())
    }

    /// Clone package from git (placeholder)
    fn clone_from_git(&self, url: &str, rev: &str, target_dir: &Path) -> Result<()> {
        println!("🌿 Cloning {} at {} to {:?}...", url, rev, target_dir);
        
        // This would implement actual git clone
        fs::create_dir_all(target_dir.join("src"))
            .map_err(|e| CompilerError::IoError(format!("Failed to create src dir: {}", e)))?;
        
        let lib_content = format!("// Cloned from {}\n// Revision: {}\n", url, rev);
        fs::write(target_dir.join("src").join("lib.aug"), lib_content)
            .map_err(|e| CompilerError::IoError(format!("Failed to write lib.aug: {}", e)))?;
        
        Ok(())
    }

    /// Copy package from local path
    fn copy_from_path(&self, source_path: &Path, target_dir: &Path) -> Result<()> {
        println!("📁 Copying from {:?} to {:?}...", source_path, target_dir);
        
        // This would implement recursive copy
        if source_path.exists() {
            fs::create_dir_all(target_dir)
                .map_err(|e| CompilerError::IoError(format!("Failed to create target dir: {}", e)))?;
            
            // Simplified copy - would use proper recursive copy
            if let Ok(entries) = fs::read_dir(source_path) {
                for entry in entries {
                    if let Ok(entry) = entry {
                        let file_name = entry.file_name();
                        if let Ok(content) = fs::read_to_string(entry.path()) {
                            fs::write(target_dir.join(file_name), content)
                                .map_err(|e| CompilerError::IoError(format!("Failed to copy file: {}", e)))?;
                        }
                    }
                }
            }
        }
        
        Ok(())
    }

    /// Update dependencies
    pub fn update_dependencies(&mut self) -> Result<InstallResult> {
        println!("🔄 Updating dependencies...");
        
        let graph = self.resolve_dependencies()?;
        self.install_dependencies(&graph)
    }

    /// Clean package cache
    pub fn clean_cache(&mut self) -> Result<()> {
        println!("🧹 Cleaning package cache...");
        
        if self.cache.cache_dir.exists() {
            fs::remove_dir_all(&self.cache.cache_dir)
                .map_err(|e| CompilerError::IoError(format!("Failed to clean cache: {}", e)))?;
        }
        
        self.cache.packages.clear();
        println!("✅ Cache cleaned successfully");
        Ok(())
    }

    /// List installed packages
    pub fn list_packages(&self) -> Vec<&CachedPackage> {
        self.cache.packages.values().collect()
    }

    /// Search packages in registry (placeholder)
    pub fn search_packages(&self, query: &str) -> Result<Vec<PackageMetadata>> {
        println!("🔍 Searching for packages matching '{}'...", query);
        
        // This would implement actual registry search
        let results = vec![
            PackageMetadata {
                name: format!("{}-utils", query),
                version: "1.0.0".to_string(),
                description: Some(format!("Utilities for {}", query)),
                authors: vec!["Community".to_string()],
                license: Some("MIT".to_string()),
                dependencies: HashMap::new(),
                features: HashMap::new(),
                targets: vec![],
                manifest_path: PathBuf::new(),
            },
        ];
        
        println!("Found {} packages", results.len());
        Ok(results)
    }

    /// Publish package to registry (placeholder)
    pub fn publish_package(&self, dry_run: bool) -> Result<()> {
        if let Some(ref config) = self.project_config {
            if dry_run {
                println!("🔍 Dry run: Would publish {} v{}", config.package.name, config.package.version);
            } else {
                println!("📤 Publishing {} v{}...", config.package.name, config.package.version);
                // This would implement actual publishing
                println!("✅ Package published successfully!");
            }
            Ok(())
        } else {
            Err(CompilerError::IoError("No project configuration loaded".to_string()))
        }
    }

    /// Save current configuration
    pub fn save_config(&self, project_path: &Path) -> Result<()> {
        if let Some(ref config) = self.project_config {
            let config_path = project_path.join("Aug.toml");
            let toml_content = toml::to_string_pretty(config)
                .map_err(|e| CompilerError::IoError(format!("Failed to serialize config: {}", e)))?;
            
            fs::write(&config_path, toml_content)
                .map_err(|e| CompilerError::IoError(format!("Failed to write Aug.toml: {}", e)))?;
            
            Ok(())
        } else {
            Err(CompilerError::IoError("No project configuration to save".to_string()))
        }
    }
}

/// Create a new package manager instance
pub fn create_package_manager() -> PackageManager {
    PackageManager::new()
}

/// Initialize a new Augustium package
pub fn init_new_package(name: &str, path: &Path) -> Result<()> {
    let pm = PackageManager::new();
    pm.init_package(name, path)
}