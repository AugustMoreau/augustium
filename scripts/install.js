#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const https = require('https');
const { execSync } = require('child_process');
const tar = require('tar');

// Configuration
const GITHUB_REPO = 'AugustMoreau/augustium';
const BINARY_DIR = path.join(__dirname, '..', 'bin');

// Platform detection
function getPlatform() {
  const platform = process.platform;
  const arch = process.arch;
  
  const platformMap = {
    'darwin': 'macos',
    'linux': 'linux',
    'win32': 'windows'
  };
  
  const archMap = {
    'x64': 'x86_64',
    'arm64': 'aarch64'
  };
  
  const mappedPlatform = platformMap[platform];
  const mappedArch = archMap[arch];
  
  if (!mappedPlatform || !mappedArch) {
    throw new Error(`Unsupported platform: ${platform}-${arch}`);
  }
  
  return `${mappedPlatform}-${mappedArch}`;
}

// Get latest release info from GitHub
function getLatestRelease() {
  return new Promise((resolve, reject) => {
    const options = {
      hostname: 'api.github.com',
      path: `/repos/${GITHUB_REPO}/releases/latest`,
      headers: {
        'User-Agent': 'augustium-npm-installer'
      }
    };
    
    https.get(options, (res) => {
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => {
        try {
          const release = JSON.parse(data);
          resolve(release);
        } catch (e) {
          reject(new Error('Failed to parse release data'));
        }
      });
    }).on('error', reject);
  });
}

// Download file
function downloadFile(url, dest) {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(dest);
    
    https.get(url, (response) => {
      if (response.statusCode === 302 || response.statusCode === 301) {
        // Handle redirect
        return downloadFile(response.headers.location, dest)
          .then(resolve)
          .catch(reject);
      }
      
      if (response.statusCode !== 200) {
        reject(new Error(`Download failed: ${response.statusCode}`));
        return;
      }
      
      response.pipe(file);
      
      file.on('finish', () => {
        file.close();
        resolve();
      });
      
      file.on('error', (err) => {
        fs.unlink(dest, () => {});
        reject(err);
      });
    }).on('error', reject);
  });
}

// Extract archive
function extractArchive(archivePath, extractDir) {
  return new Promise((resolve, reject) => {
    if (archivePath.endsWith('.tar.gz')) {
      tar.extract({
        file: archivePath,
        cwd: extractDir,
        strip: 1
      }).then(resolve).catch(reject);
    } else {
      reject(new Error('Unsupported archive format'));
    }
  });
}

// Make binary executable
function makeExecutable(filePath) {
  if (process.platform !== 'win32') {
    try {
      execSync(`chmod +x "${filePath}"`);
    } catch (e) {
      console.warn('Warning: Could not make binary executable');
    }
  }
}

// Main installation function
async function install() {
  try {
    console.log('🚀 Installing Augustium...');
    
    // Detect platform
    const platform = getPlatform();
    console.log(`📱 Detected platform: ${platform}`);
    
    // Get latest release
    console.log('📡 Fetching latest release...');
    const release = await getLatestRelease();
    
    // Find appropriate asset
    const assetName = `augustium-${platform}.tar.gz`;
    const asset = release.assets.find(a => a.name === assetName);
    
    if (!asset) {
      // Fallback: try to build from source if Rust is available
      console.log('⚠️  Pre-built binary not available for your platform.');
      console.log('🔨 Attempting to build from source...');
      
      try {
        execSync('cargo --version', { stdio: 'ignore' });
        console.log('✅ Rust found, building from source...');
        
        // Clone and build
        const tempDir = path.join(__dirname, '..', 'temp');
        fs.mkdirSync(tempDir, { recursive: true });
        
        execSync(`git clone https://github.com/${GITHUB_REPO}.git "${tempDir}"`, { stdio: 'inherit' });
        execSync('cargo build --release', { cwd: tempDir, stdio: 'inherit' });
        
        // Copy binaries
        fs.mkdirSync(BINARY_DIR, { recursive: true });
        const targetDir = path.join(tempDir, 'target', 'release');
        
        const binaries = ['augustc', 'august'];
        for (const binary of binaries) {
          const ext = process.platform === 'win32' ? '.exe' : '';
          const src = path.join(targetDir, binary + ext);
          const dest = path.join(BINARY_DIR, binary + ext);
          
          if (fs.existsSync(src)) {
            fs.copyFileSync(src, dest);
            makeExecutable(dest);
            console.log(`✅ Installed: ${binary}`);
          }
        }
        
        // Cleanup
        fs.rmSync(tempDir, { recursive: true, force: true });
        
      } catch (buildError) {
        throw new Error(`Build from source failed. Please install Rust and try again, or download pre-built binaries from: https://github.com/${GITHUB_REPO}/releases`);
      }
      
      return;
    }
    
    // Download pre-built binary
    console.log(`📥 Downloading ${asset.name}...`);
    
    fs.mkdirSync(BINARY_DIR, { recursive: true });
    const archivePath = path.join(BINARY_DIR, asset.name);
    
    await downloadFile(asset.browser_download_url, archivePath);
    
    // Extract
    console.log('📦 Extracting archive...');
    await extractArchive(archivePath, BINARY_DIR);
    
    // Make binaries executable
    const binaries = ['augustc', 'august'];
    for (const binary of binaries) {
      const ext = process.platform === 'win32' ? '.exe' : '';
      const binaryPath = path.join(BINARY_DIR, binary + ext);
      
      if (fs.existsSync(binaryPath)) {
        makeExecutable(binaryPath);
        console.log(`✅ Installed: ${binary}`);
      }
    }
    
    // Cleanup
    fs.unlinkSync(archivePath);
    
    console.log('🎉 Augustium installed successfully!');
    console.log('');
    console.log('Try running:');
    console.log('  augustc --version');
    console.log('  august --help');
    
  } catch (error) {
    console.error('❌ Installation failed:', error.message);
    process.exit(1);
  }
}

// Run installation
if (require.main === module) {
  install();
}

module.exports = { install };