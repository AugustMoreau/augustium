//! IDE Plugin Support for Augustium
//!
//! This module provides IDE integration capabilities, including plugin generators
//! for popular development environments like VS Code, IntelliJ IDEA, Vim/Neovim,
//! Emacs, and Sublime Text.

use crate::error::CompilerError;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};

/// Supported IDE types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum IdeType {
    VSCode,
    IntelliJ,
    Vim,
    Neovim,
    Emacs,
    SublimeText,
    Atom,
    Custom(String),
}

impl IdeType {
    /// Get the IDE name as a string
    pub fn name(&self) -> &str {
        match self {
            IdeType::VSCode => "Visual Studio Code",
            IdeType::IntelliJ => "IntelliJ IDEA",
            IdeType::Vim => "Vim",
            IdeType::Neovim => "Neovim",
            IdeType::Emacs => "Emacs",
            IdeType::SublimeText => "Sublime Text",
            IdeType::Atom => "Atom",
            IdeType::Custom(name) => name,
        }
    }
    
    /// Get the configuration directory for the IDE
    #[allow(dead_code)]
    pub fn config_dir(&self) -> &str {
        match self {
            IdeType::VSCode => ".vscode",
            IdeType::IntelliJ => ".idea",
            IdeType::Vim => ".vim",
            IdeType::Neovim => ".config/nvim",
            IdeType::Emacs => ".emacs.d",
            IdeType::SublimeText => ".sublime",
            IdeType::Atom => ".atom",
            IdeType::Custom(_) => ".custom",
        }
    }
}

/// Plugin configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginConfig {
    pub ide_type: IdeType,
    pub plugin_name: String,
    pub version: String,
    pub description: String,
    pub author: String,
    pub features: Vec<PluginFeature>,
    pub dependencies: Vec<String>,
    pub settings: HashMap<String, serde_json::Value>,
}

/// Plugin features
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PluginFeature {
    SyntaxHighlighting,
    AutoCompletion,
    ErrorChecking,
    Debugging,
    Formatting,
    Snippets,
    ProjectTemplates,
    BuildIntegration,
    TestRunner,
    Documentation,
    Refactoring,
    Navigation,
}

/// Plugin file structure
#[derive(Debug, Clone)]
pub struct PluginFile {
    pub path: PathBuf,
    pub content: String,
    #[allow(dead_code)]
    pub file_type: PluginFileType,
}

/// Plugin file types
#[derive(Debug, Clone, PartialEq)]
pub enum PluginFileType {
    Manifest,
    Grammar,
    Snippets,
    Configuration,
    Script,
    #[allow(dead_code)]
    Theme,
    #[allow(dead_code)]
    Icon,
    Documentation,
}

/// IDE plugin generator
pub struct IdePluginGenerator {
    config: PluginConfig,
    output_dir: PathBuf,
    files: Vec<PluginFile>,
}

impl IdePluginGenerator {
    /// Create a new plugin generator
    pub fn new(config: PluginConfig, output_dir: PathBuf) -> Self {
        Self {
            config,
            output_dir,
            files: Vec::new(),
        }
    }
    
    /// Generate plugin files for the specified IDE
    pub fn generate(&mut self) -> Result<(), CompilerError> {
        match self.config.ide_type {
            IdeType::VSCode => self.generate_vscode_plugin()?,
            IdeType::IntelliJ => self.generate_intellij_plugin()?,
            IdeType::Vim => self.generate_vim_plugin()?,
            IdeType::Neovim => self.generate_neovim_plugin()?,
            IdeType::Emacs => self.generate_emacs_plugin()?,
            IdeType::SublimeText => self.generate_sublime_plugin()?,
            IdeType::Atom => self.generate_atom_plugin()?,
            IdeType::Custom(_) => self.generate_custom_plugin()?,
        }
        
        self.write_files()?;
        Ok(())
    }
    
    /// Generate VS Code plugin
    fn generate_vscode_plugin(&mut self) -> Result<(), CompilerError> {
        // Package.json
        let package_json = self.create_vscode_package_json();
        self.add_file("package.json", package_json, PluginFileType::Manifest);
        
        // Language configuration
        let language_config = self.create_vscode_language_config();
        self.add_file("language-configuration.json", language_config, PluginFileType::Configuration);
        
        // TextMate grammar
        let grammar = self.create_textmate_grammar();
        self.add_file("syntaxes/augustium.tmLanguage.json", grammar, PluginFileType::Grammar);
        
        // Snippets
        if self.config.features.contains(&PluginFeature::Snippets) {
            let snippets = self.create_vscode_snippets();
            self.add_file("snippets/augustium.json", snippets, PluginFileType::Snippets);
        }
        
        // Extension main file
        let extension_main = self.create_vscode_extension_main();
        self.add_file("src/extension.ts", extension_main, PluginFileType::Script);
        
        // README
        let readme = self.create_readme();
        self.add_file("README.md", readme, PluginFileType::Documentation);
        
        Ok(())
    }
    
    /// Generate IntelliJ plugin
    fn generate_intellij_plugin(&mut self) -> Result<(), CompilerError> {
        // Plugin.xml
        let plugin_xml = self.create_intellij_plugin_xml();
        self.add_file("src/main/resources/META-INF/plugin.xml", plugin_xml, PluginFileType::Manifest);
        
        // Language definition
        let language_def = self.create_intellij_language_definition();
        self.add_file("src/main/java/com/augustium/AugustiumLanguage.java", language_def, PluginFileType::Script);
        
        // Lexer
        let lexer = self.create_intellij_lexer();
        self.add_file("src/main/java/com/augustium/AugustiumLexer.java", lexer, PluginFileType::Script);
        
        // Parser
        let parser = self.create_intellij_parser();
        self.add_file("src/main/java/com/augustium/AugustiumParser.java", parser, PluginFileType::Script);
        
        // Build script
        let build_gradle = self.create_intellij_build_gradle();
        self.add_file("build.gradle", build_gradle, PluginFileType::Configuration);
        
        Ok(())
    }
    
    /// Generate Vim plugin
    fn generate_vim_plugin(&mut self) -> Result<(), CompilerError> {
        // Syntax file
        let syntax = self.create_vim_syntax();
        self.add_file("syntax/augustium.vim", syntax, PluginFileType::Grammar);
        
        // Filetype detection
        let filetype = self.create_vim_filetype();
        self.add_file("ftdetect/augustium.vim", filetype, PluginFileType::Configuration);
        
        // Indentation
        let indent = self.create_vim_indent();
        self.add_file("indent/augustium.vim", indent, PluginFileType::Configuration);
        
        // Plugin main file
        let plugin_main = self.create_vim_plugin_main();
        self.add_file("plugin/augustium.vim", plugin_main, PluginFileType::Script);
        
        Ok(())
    }
    
    /// Generate Neovim plugin
    fn generate_neovim_plugin(&mut self) -> Result<(), CompilerError> {
        // Lua configuration
        let lua_config = self.create_neovim_lua_config();
        self.add_file("lua/augustium/init.lua", lua_config, PluginFileType::Script);
        
        // Tree-sitter grammar
        let treesitter = self.create_treesitter_grammar();
        self.add_file("queries/augustium/highlights.scm", treesitter, PluginFileType::Grammar);
        
        // LSP configuration
        let lsp_config = self.create_neovim_lsp_config();
        self.add_file("lua/augustium/lsp.lua", lsp_config, PluginFileType::Configuration);
        
        Ok(())
    }
    
    /// Generate Emacs plugin
    fn generate_emacs_plugin(&mut self) -> Result<(), CompilerError> {
        // Major mode
        let major_mode = self.create_emacs_major_mode();
        self.add_file("augustium-mode.el", major_mode, PluginFileType::Script);
        
        // Package definition
        let package_def = self.create_emacs_package_definition();
        self.add_file("augustium-mode-pkg.el", package_def, PluginFileType::Manifest);
        
        Ok(())
    }
    
    /// Generate Sublime Text plugin
    fn generate_sublime_plugin(&mut self) -> Result<(), CompilerError> {
        // Syntax definition
        let syntax = self.create_sublime_syntax();
        self.add_file("Augustium.sublime-syntax", syntax, PluginFileType::Grammar);
        
        // Completions
        let completions = self.create_sublime_completions();
        self.add_file("Augustium.sublime-completions", completions, PluginFileType::Snippets);
        
        // Build system
        let build_system = self.create_sublime_build_system();
        self.add_file("Augustium.sublime-build", build_system, PluginFileType::Configuration);
        
        Ok(())
    }
    
    /// Generate Atom plugin
    fn generate_atom_plugin(&mut self) -> Result<(), CompilerError> {
        // Package.json
        let package_json = self.create_atom_package_json();
        self.add_file("package.json", package_json, PluginFileType::Manifest);
        
        // Grammar
        let grammar = self.create_atom_grammar();
        self.add_file("grammars/augustium.cson", grammar, PluginFileType::Grammar);
        
        // Snippets
        let snippets = self.create_atom_snippets();
        self.add_file("snippets/augustium.cson", snippets, PluginFileType::Snippets);
        
        Ok(())
    }
    
    /// Generate custom plugin
    fn generate_custom_plugin(&mut self) -> Result<(), CompilerError> {
        // Basic configuration
        let config = self.create_generic_config();
        self.add_file("config.json", config, PluginFileType::Configuration);
        
        // Generic syntax highlighting
        let syntax = self.create_generic_syntax();
        self.add_file("syntax.json", syntax, PluginFileType::Grammar);
        
        Ok(())
    }
    
    /// Add a file to the plugin
    fn add_file(&mut self, path: &str, content: String, file_type: PluginFileType) {
        self.files.push(PluginFile {
            path: PathBuf::from(path),
            content,
            file_type,
        });
    }
    
    /// Write all plugin files to disk
    fn write_files(&self) -> Result<(), CompilerError> {
        for file in &self.files {
            let full_path = self.output_dir.join(&file.path);
            
            // Create parent directories
            if let Some(parent) = full_path.parent() {
                fs::create_dir_all(parent).map_err(|e| CompilerError::IoError(
                    format!("Failed to create directory: {}", e)
                ))?;
            }
            
            // Write file
            fs::write(&full_path, &file.content).map_err(|e| CompilerError::IoError(
                format!("Failed to write file {}: {}", full_path.display(), e)
            ))?;
        }
        
        Ok(())
    }
    
    // VS Code specific generators
    fn create_vscode_package_json(&self) -> String {
        format!(r#"{{
    "name": "augustium",
    "displayName": "Augustium Language Support",
    "description": "Language support for Augustium smart contract development",
    "version": "{}",
    "publisher": "{}",
    "engines": {{
        "vscode": "^1.60.0"
    }},
    "categories": [
        "Programming Languages",
        "Snippets",
        "Debuggers"
    ],
    "contributes": {{
        "languages": [{{
            "id": "augustium",
            "aliases": ["Augustium", "augustium"],
            "extensions": [".aug"],
            "configuration": "./language-configuration.json"
        }}],
        "grammars": [{{
            "language": "augustium",
            "scopeName": "source.augustium",
            "path": "./syntaxes/augustium.tmLanguage.json"
        }}],
        "snippets": [{{
            "language": "augustium",
            "path": "./snippets/augustium.json"
        }}]
    }},
    "main": "./out/extension.js",
    "scripts": {{
        "vscode:prepublish": "npm run compile",
        "compile": "tsc -p ./",
        "watch": "tsc -watch -p ./"
    }},
    "devDependencies": {{
        "@types/vscode": "^1.60.0",
        "typescript": "^4.4.0"
    }}
}}
"#, self.config.version, self.config.author)
    }
    
    fn create_vscode_language_config(&self) -> String {
        r#"{
    "comments": {
        "lineComment": "//",
        "blockComment": ["/*", "*/"]
    },
    "brackets": [
        ["{", "}"],
        ["[", "]"],
        ["(", ")"]
    ],
    "autoClosingPairs": [
        {"open": "{", "close": "}"},
        {"open": "[", "close": "]"},
        {"open": "(", "close": ")"},
        {"open": "'", "close": "'"},
        {"open": "\"", "close": "\""}
    ],
    "surroundingPairs": [
        ["{", "}"],
        ["[", "]"],
        ["(", ")"],
        ["'", "'"],
        ["\"", "\""]
    ]
}
"#.to_string()
    }
    
    fn create_textmate_grammar(&self) -> String {
        serde_json::to_string_pretty(&serde_json::json!({
            "$schema": "https://raw.githubusercontent.com/martinring/tmlanguage/master/tmlanguage.json",
            "name": "Augustium",
            "patterns": [
                {"include": "#keywords"},
                {"include": "#strings"},
                {"include": "#comments"},
                {"include": "#numbers"},
                {"include": "#types"},
                {"include": "#functions"}
            ],
            "repository": {
                "keywords": {
                    "patterns": [{
                        "name": "keyword.control.augustium",
                        "match": "\\b(if|else|while|for|return|break|continue|contract|function|event|modifier|require|assert|revert)\\b"
                    }]
                },
                "strings": {
                    "name": "string.quoted.double.augustium",
                    "begin": "\"",
                    "end": "\"",
                    "patterns": [{
                        "name": "constant.character.escape.augustium",
                        "match": "\\\\."
                    }]
                },
                "comments": {
                    "patterns": [
                        {
                            "name": "comment.line.double-slash.augustium",
                            "match": "//.*$"
                        },
                        {
                            "name": "comment.block.augustium",
                            "begin": "/\\*",
                            "end": "\\*/"
                        }
                    ]
                },
                "numbers": {
                    "name": "constant.numeric.augustium",
                    "match": "\\b\\d+(\\.\\d+)?\\b"
                },
                "types": {
                    "name": "storage.type.augustium",
                    "match": "\\b(uint|int|bool|address|string|bytes|mapping)\\b"
                },
                "functions": {
                    "name": "entity.name.function.augustium",
                    "match": "\\b[a-zA-Z_][a-zA-Z0-9_]*(?=\\s*\\()"
                }
            },
            "scopeName": "source.augustium"
        })).unwrap_or_default()
    }
    
    fn create_vscode_snippets(&self) -> String {
        r#"{
    "Contract": {
        "prefix": "contract",
        "body": [
            "contract ${1:ContractName} {",
            "    ${2:// Contract body}",
            "}"
        ],
        "description": "Create a new contract"
    },
    "Function": {
        "prefix": "function",
        "body": [
            "function ${1:functionName}(${2:parameters}) ${3:public} ${4:returns (${5:returnType})} {",
            "    ${6:// Function body}",
            "}"
        ],
        "description": "Create a new function"
    },
    "Event": {
        "prefix": "event",
        "body": [
            "event ${1:EventName}(${2:parameters});"
        ],
        "description": "Create a new event"
    },
    "Modifier": {
        "prefix": "modifier",
        "body": [
            "modifier ${1:modifierName}(${2:parameters}) {",
            "    ${3:// Modifier logic}",
            "    _;",
            "}"
        ],
        "description": "Create a new modifier"
    }
}
"#.to_string()
    }
    
    fn create_vscode_extension_main(&self) -> String {
        r#"import * as vscode from 'vscode';

export function activate(context: vscode.ExtensionContext) {
    console.log('Augustium extension is now active!');
    
    // Register language server if available
    // TODO: Implement LSP client
    
    // Register commands
    let disposable = vscode.commands.registerCommand('augustium.compile', () => {
        vscode.window.showInformationMessage('Compiling Augustium contract...');
        // TODO: Implement compilation
    });
    
    context.subscriptions.push(disposable);
}

export function deactivate() {}
"#.to_string()
    }
    
    // IntelliJ specific generators
    fn create_intellij_plugin_xml(&self) -> String {
        format!(r#"<idea-plugin>
    <id>com.augustium.plugin</id>
    <name>Augustium Language Support</name>
    <version>{}</version>
    <vendor email="{}" url="https://augustium.org">{}</vendor>
    
    <description><![CDATA[
        Language support for Augustium smart contract development.
        Features include syntax highlighting, code completion, and error checking.
    ]]></description>
    
    <idea-version since-build="203"/>
    
    <extensions defaultExtensionNs="com.intellij">
        <fileType name="Augustium" implementationClass="com.augustium.AugustiumFileType" fieldName="INSTANCE" language="Augustium" extensions="aug"/>
        <lang.parserDefinition language="Augustium" implementationClass="com.augustium.AugustiumParserDefinition"/>
        <lang.syntaxHighlighterFactory language="Augustium" implementationClass="com.augustium.AugustiumSyntaxHighlighterFactory"/>
    </extensions>
</idea-plugin>
"#, self.config.version, self.config.author, self.config.author)
    }
    
    fn create_intellij_language_definition(&self) -> String {
        r#"package com.augustium;

import com.intellij.lang.Language;

public class AugustiumLanguage extends Language {
    public static final AugustiumLanguage INSTANCE = new AugustiumLanguage();
    
    private AugustiumLanguage() {
        super("Augustium");
    }
}
"#.to_string()
    }
    
    fn create_intellij_lexer(&self) -> String {
        r#"package com.augustium;

import com.intellij.lexer.FlexLexer;
import com.intellij.psi.tree.IElementType;

public class AugustiumLexer implements FlexLexer {
    // TODO: Implement lexer logic
    
    @Override
    public void yybegin(int state) {
        // Implementation
    }
    
    @Override
    public int yystate() {
        return 0;
    }
    
    @Override
    public IElementType getTokenType() {
        return null;
    }
    
    @Override
    public int getTokenStart() {
        return 0;
    }
    
    @Override
    public int getTokenEnd() {
        return 0;
    }
    
    @Override
    public void advance() throws java.io.IOException {
        // Implementation
    }
    
    @Override
    public void reset(CharSequence buffer, int start, int end, int initialState) {
        // Implementation
    }
}
"#.to_string()
    }
    
    fn create_intellij_parser(&self) -> String {
        r#"package com.augustium;

import com.intellij.lang.ASTNode;
import com.intellij.lang.PsiBuilder;
import com.intellij.lang.PsiParser;
import com.intellij.psi.tree.IElementType;
import org.jetbrains.annotations.NotNull;

public class AugustiumParser implements PsiParser {
    @NotNull
    @Override
    public ASTNode parse(@NotNull IElementType root, @NotNull PsiBuilder builder) {
        // TODO: Implement parser logic
        PsiBuilder.Marker rootMarker = builder.mark();
        
        // Parse the file
        while (!builder.eof()) {
            builder.advanceLexer();
        }
        
        rootMarker.done(root);
        return builder.getTreeBuilt();
    }
}
"#.to_string()
    }
    
    fn create_intellij_build_gradle(&self) -> String {
        r#"plugins {
    id 'org.jetbrains.intellij' version '1.0'
    id 'java'
}

group 'com.augustium'
version '1.0.0'

repositories {
    mavenCentral()
}

intellij {
    version '2021.3'
}

patchPluginXml {
    changeNotes """
        Initial release of Augustium language support.
    """
}
"#.to_string()
    }
    
    // Vim specific generators
    fn create_vim_syntax(&self) -> String {
        r#"" Vim syntax file
" Language: Augustium
" Maintainer: Augustium Team

if exists("b:current_syntax")
    finish
endif

" Keywords
syn keyword augKeyword contract function event modifier require assert revert
syn keyword augConditional if else
syn keyword augRepeat while for
syn keyword augStatement return break continue
syn keyword augType uint int bool address string bytes mapping

" Comments
syn match augComment "//.*$"
syn region augComment start="/\*" end="\*/"

" Strings
syn region augString start='"' end='"' skip='\\"'

" Numbers
syn match augNumber "\d\+"
syn match augFloat "\d\+\.\d\+"

" Highlighting
hi def link augKeyword Keyword
hi def link augConditional Conditional
hi def link augRepeat Repeat
hi def link augStatement Statement
hi def link augType Type
hi def link augComment Comment
hi def link augString String
hi def link augNumber Number
hi def link augFloat Float

let b:current_syntax = "augustium"
"#.to_string()
    }
    
    fn create_vim_filetype(&self) -> String {
        r#"" Filetype detection for Augustium
au BufRead,BufNewFile *.aug set filetype=augustium
"#.to_string()
    }
    
    fn create_vim_indent(&self) -> String {
        r#"" Vim indent file
" Language: Augustium

if exists("b:did_indent")
    finish
endif
let b:did_indent = 1

setlocal indentexpr=GetAugustiumIndent()
setlocal indentkeys+=0{,0},0),0],!^F,o,O,e
setlocal autoindent
setlocal smartindent

function! GetAugustiumIndent()
    let line = getline(v:lnum)
    let previousNum = prevnonblank(v:lnum - 1)
    let previous = getline(previousNum)
    
    if previous =~ '{\s*$'
        return indent(previousNum) + &shiftwidth
    elseif line =~ '^\s*}'
        return indent(previousNum) - &shiftwidth
    else
        return indent(previousNum)
    endif
endfunction
"#.to_string()
    }
    
    fn create_vim_plugin_main(&self) -> String {
        r#"" Augustium Vim plugin
" Main plugin file

if exists('g:loaded_augustium')
    finish
endif
let g:loaded_augustium = 1

" Commands
command! AugustiumCompile call augustium#compile#run()
command! AugustiumFormat call augustium#format#run()

" Mappings
nnoremap <leader>ac :AugustiumCompile<CR>
nnoremap <leader>af :AugustiumFormat<CR>
"#.to_string()
    }
    
    // Neovim specific generators
    fn create_neovim_lua_config(&self) -> String {
        r#"-- Augustium Neovim plugin
local M = {}

-- Setup function
function M.setup(opts)
    opts = opts or {}
    
    -- Set up filetype detection
    vim.filetype.add({
        extension = {
            aug = 'augustium',
        },
    })
    
    -- Set up LSP if available
    if opts.lsp then
        require('augustium.lsp').setup(opts.lsp)
    end
end

-- Compile function
function M.compile()
    local file = vim.fn.expand('%')
    if vim.fn.fnamemodify(file, ':e') == 'aug' then
        vim.cmd('!augustc compile ' .. file)
    else
        print('Not an Augustium file')
    end
end

-- Format function
function M.format()
    local file = vim.fn.expand('%')
    if vim.fn.fnamemodify(file, ':e') == 'aug' then
        vim.cmd('!augustc fmt ' .. file)
        vim.cmd('edit!')
    else
        print('Not an Augustium file')
    end
end

return M
"#.to_string()
    }
    
    fn create_treesitter_grammar(&self) -> String {
        r#"; Augustium Tree-sitter highlights

(comment) @comment

(string_literal) @string
(number_literal) @number
(boolean_literal) @boolean

(identifier) @variable
(type_identifier) @type

(function_definition
  name: (identifier) @function)

(contract_definition
  name: (identifier) @type.definition)

(event_definition
  name: (identifier) @function.special)

[
  "contract"
  "function"
  "event"
  "modifier"
  "if"
  "else"
  "while"
  "for"
  "return"
  "break"
  "continue"
] @keyword

[
  "require"
  "assert"
  "revert"
] @function.builtin

[
  "uint"
  "int"
  "bool"
  "address"
  "string"
  "bytes"
  "mapping"
] @type.builtin

[
  "{"
  "}"
  "["
  "]"
  "("
  ")"
] @punctuation.bracket

[
  ";"
  ","
  "."
] @punctuation.delimiter

[
  "="
  "+"
  "-"
  "*"
  "/"
  "%"
  "=="
  "!="
  "<"
  ">"
  "<="
  ">="
  "&&"
  "||"
  "!"
] @operator
"#.to_string()
    }
    
    fn create_neovim_lsp_config(&self) -> String {
        r#"-- Augustium LSP configuration for Neovim
local M = {}

function M.setup(opts)
    opts = opts or {}
    
    local lspconfig = require('lspconfig')
    
    -- Configure Augustium LSP
    lspconfig.augustium_lsp = {
        default_config = {
            cmd = { 'augustc', 'lsp' },
            filetypes = { 'augustium' },
            root_dir = function(fname)
                return lspconfig.util.find_git_ancestor(fname) or
                       lspconfig.util.path.dirname(fname)
            end,
            settings = {},
        },
    }
    
    lspconfig.augustium_lsp.setup(opts)
end

return M
"#.to_string()
    }
    
    // Emacs specific generators
    fn create_emacs_major_mode(&self) -> String {
        r#";;; augustium-mode.el --- Major mode for Augustium smart contracts

;;; Commentary:
;; This package provides a major mode for editing Augustium smart contract files.

;;; Code:

(defgroup augustium nil
  "Major mode for editing Augustium smart contracts."
  :group 'languages)

(defcustom augustium-indent-offset 4
  "Indentation offset for Augustium code."
  :type 'integer
  :group 'augustium)

(defvar augustium-keywords
  '("contract" "function" "event" "modifier" "if" "else" "while" "for"
    "return" "break" "continue" "require" "assert" "revert")
  "Augustium keywords.")

(defvar augustium-types
  '("uint" "int" "bool" "address" "string" "bytes" "mapping")
  "Augustium types.")

(defvar augustium-font-lock-keywords
  `((,(regexp-opt augustium-keywords 'words) . font-lock-keyword-face)
    (,(regexp-opt augustium-types 'words) . font-lock-type-face)
    ("\\b[0-9]+\\b" . font-lock-constant-face)
    ("\".*?\"" . font-lock-string-face)
    ("//.*$" . font-lock-comment-face)
    ("/\\*.*?\\*/" . font-lock-comment-face))
  "Font lock keywords for Augustium mode.")

(defvar augustium-mode-syntax-table
  (let ((table (make-syntax-table)))
    (modify-syntax-entry ?/ ". 124b" table)
    (modify-syntax-entry ?* ". 23" table)
    (modify-syntax-entry ?\n "> b" table)
    (modify-syntax-entry ?{ "(}" table)
    (modify-syntax-entry ?} "){" table)
    (modify-syntax-entry ?[ "(]" table)
    (modify-syntax-entry ?] ")[" table)
    (modify-syntax-entry ?( "()" table)
    (modify-syntax-entry ?) ")(" table)
    table)
  "Syntax table for Augustium mode.")

(defun augustium-indent-line ()
  "Indent current line as Augustium code."
  (interactive)
  (let ((indent-col 0))
    (save-excursion
      (beginning-of-line)
      (when (not (bobp))
        (forward-line -1)
        (setq indent-col (current-indentation))
        (when (looking-at ".*{\\s-*$")
          (setq indent-col (+ indent-col augustium-indent-offset)))))
    (save-excursion
      (beginning-of-line)
      (when (looking-at "\\s-*}")
        (setq indent-col (- indent-col augustium-indent-offset))))
    (indent-line-to (max 0 indent-col))))

;;;###autoload
(define-derived-mode augustium-mode prog-mode "Augustium"
  "Major mode for editing Augustium smart contracts."
  (setq font-lock-defaults '(augustium-font-lock-keywords))
  (setq indent-line-function 'augustium-indent-line)
  (setq comment-start "// ")
  (setq comment-end "")
  (setq comment-start-skip "//+\\s-*"))

;;;###autoload
(add-to-list 'auto-mode-alist '("\\.aug\\'" . augustium-mode))

(provide 'augustium-mode)

;;; augustium-mode.el ends here
"#.to_string()
    }
    
    fn create_emacs_package_definition(&self) -> String {
        format!(r#";;; augustium-mode-pkg.el --- Package definition for augustium-mode

(define-package "augustium-mode" "{}"
  "Major mode for Augustium smart contracts"
  '((emacs "24.3"))
  :keywords '("languages" "augustium" "smart-contracts")
  :url "https://augustium.org")

;;; augustium-mode-pkg.el ends here
"#, self.config.version)
    }
    
    // Sublime Text specific generators
    fn create_sublime_syntax(&self) -> String {
        r#"%YAML 1.2
---
name: Augustium
file_extensions:
  - aug
scope: source.augustium

contexts:
  main:
    - include: comments
    - include: strings
    - include: numbers
    - include: keywords
    - include: types
    - include: functions

  comments:
    - match: '//'
      scope: punctuation.definition.comment.augustium
      push:
        - meta_scope: comment.line.double-slash.augustium
        - match: $
          pop: true
    - match: '/\*'
      scope: punctuation.definition.comment.augustium
      push:
        - meta_scope: comment.block.augustium
        - match: '\*/'
          scope: punctuation.definition.comment.augustium
          pop: true

  strings:
    - match: '"'
      scope: punctuation.definition.string.begin.augustium
      push:
        - meta_scope: string.quoted.double.augustium
        - match: '\\\\.|\\\\"'
          scope: constant.character.escape.augustium
        - match: '"'
          scope: punctuation.definition.string.end.augustium
          pop: true

  numbers:
    - match: '\b\d+(\.\d+)?\b'
      scope: constant.numeric.augustium

  keywords:
    - match: '\b(contract|function|event|modifier|if|else|while|for|return|break|continue|require|assert|revert)\b'
      scope: keyword.control.augustium

  types:
    - match: '\b(uint|int|bool|address|string|bytes|mapping)\b'
      scope: storage.type.augustium

  functions:
    - match: '\b[a-zA-Z_][a-zA-Z0-9_]*(?=\s*\()'
      scope: entity.name.function.augustium
"#.to_string()
    }
    
    fn create_sublime_completions(&self) -> String {
        r#"{
    "scope": "source.augustium",
    "completions": [
        {
            "trigger": "contract",
            "contents": "contract ${1:ContractName} {\n\t${2:// Contract body}\n}"
        },
        {
            "trigger": "function",
            "contents": "function ${1:functionName}(${2:parameters}) ${3:public} ${4:returns (${5:returnType})} {\n\t${6:// Function body}\n}"
        },
        {
            "trigger": "event",
            "contents": "event ${1:EventName}(${2:parameters});"
        },
        {
            "trigger": "modifier",
            "contents": "modifier ${1:modifierName}(${2:parameters}) {\n\t${3:// Modifier logic}\n\t_;\n}"
        },
        {
            "trigger": "require",
            "contents": "require(${1:condition}, \"${2:error message}\");"
        },
        {
            "trigger": "mapping",
            "contents": "mapping(${1:keyType} => ${2:valueType}) ${3:mappingName};"
        }
    ]
}
"#.to_string()
    }
    
    fn create_sublime_build_system(&self) -> String {
        r#"{
    "name": "Augustium",
    "cmd": ["augustc", "compile", "$file"],
    "file_regex": "^(.+):(\d+):(\d+): (.+)$",
    "selector": "source.augustium",
    "variants": [
        {
            "name": "Build and Run",
            "cmd": ["augustc", "compile", "$file", "--run"]
        },
        {
            "name": "Check Syntax",
            "cmd": ["augustc", "compile", "$file", "--check"]
        },
        {
            "name": "Format",
            "cmd": ["augustc", "fmt", "$file"]
        }
    ]
}
"#.to_string()
    }
    
    // Atom specific generators
    fn create_atom_package_json(&self) -> String {
        format!(r#"{{
    "name": "language-augustium",
    "version": "{}",
    "description": "Augustium language support for Atom",
    "keywords": ["augustium", "smart-contracts", "blockchain"],
    "repository": "https://github.com/augustium/atom-augustium",
    "license": "MIT",
    "engines": {{
        "atom": ">=1.0.0 <2.0.0"
    }}
}}
"#, self.config.version)
    }
    
    fn create_atom_grammar(&self) -> String {
        r#"'scopeName': 'source.augustium'
'name': 'Augustium'
'fileTypes': ['aug']

'patterns': [
  {
    'include': '#comments'
  }
  {
    'include': '#strings'
  }
  {
    'include': '#keywords'
  }
  {
    'include': '#types'
  }
  {
    'include': '#numbers'
  }
]

'repository':
  'comments':
    'patterns': [
      {
        'match': '//.*$'
        'name': 'comment.line.double-slash.augustium'
      }
      {
        'begin': '/\\*'
        'end': '\\*/'
        'name': 'comment.block.augustium'
      }
    ]
  
  'strings':
    'begin': '"'
    'end': '"'
    'name': 'string.quoted.double.augustium'
    'patterns': [
      {
        'match': '\\\\.|\\\\"'
        'name': 'constant.character.escape.augustium'
      }
    ]
  
  'keywords':
    'match': '\\b(contract|function|event|modifier|if|else|while|for|return|break|continue|require|assert|revert)\\b'
    'name': 'keyword.control.augustium'
  
  'types':
    'match': '\\b(uint|int|bool|address|string|bytes|mapping)\\b'
    'name': 'storage.type.augustium'
  
  'numbers':
    'match': '\\b\\d+(\\.\\d+)?\\b'
    'name': 'constant.numeric.augustium'
"#.to_string()
    }
    
    fn create_atom_snippets(&self) -> String {
        r#"'.source.augustium':
  'Contract':
    'prefix': 'contract'
    'body': 'contract ${1:ContractName} {\n\t${2:// Contract body}\n}'
  
  'Function':
    'prefix': 'function'
    'body': 'function ${1:functionName}(${2:parameters}) ${3:public} ${4:returns (${5:returnType})} {\n\t${6:// Function body}\n}'
  
  'Event':
    'prefix': 'event'
    'body': 'event ${1:EventName}(${2:parameters});'
  
  'Modifier':
    'prefix': 'modifier'
    'body': 'modifier ${1:modifierName}(${2:parameters}) {\n\t${3:// Modifier logic}\n\t_;\n}'
  
  'Require':
    'prefix': 'require'
    'body': 'require(${1:condition}, "${2:error message}");'
  
  'Mapping':
    'prefix': 'mapping'
    'body': 'mapping(${1:keyType} => ${2:valueType}) ${3:mappingName};'
"#.to_string()
    }
    
    // Generic generators
    fn create_generic_config(&self) -> String {
        serde_json::to_string_pretty(&self.config).unwrap_or_default()
    }
    
    fn create_generic_syntax(&self) -> String {
        r#"{
    "name": "Augustium",
    "extensions": [".aug"],
    "keywords": [
        "contract", "function", "event", "modifier",
        "if", "else", "while", "for",
        "return", "break", "continue",
        "require", "assert", "revert"
    ],
    "types": [
        "uint", "int", "bool", "address",
        "string", "bytes", "mapping"
    ],
    "operators": [
        "=", "+", "-", "*", "/", "%",
        "==", "!=", "<", ">", "<=", ">=",
        "&&", "||", "!"
    ],
    "brackets": [
        ["{", "}"],
        ["[", "]"],
        ["(", ")"]
    ],
    "comments": {
        "line": "//",
        "block": ["/*", "*/"]
    }
}
"#.to_string()
    }
    
    fn create_readme(&self) -> String {
        format!(r#"# Augustium Language Support

This extension provides language support for Augustium smart contract development.

## Features

- Syntax highlighting
- Code completion
- Error checking
- Code formatting
- Snippets
- Build integration

## Installation

1. Install the extension from the marketplace
2. Open an Augustium file (.aug)
3. Enjoy enhanced development experience!

## Requirements

- Augustium compiler (augustc) installed and available in PATH

## Extension Settings

This extension contributes the following settings:

- `augustium.compiler.path`: Path to the Augustium compiler
- `augustium.lsp.enabled`: Enable Language Server Protocol support
- `augustium.format.onSave`: Format code on save

## Known Issues

- LSP support is experimental
- Some advanced features may not work in all environments

## Release Notes

### {}

Initial release of Augustium language support.

## Contributing

Contributions are welcome! Please see our [contributing guidelines](CONTRIBUTING.md).

## License

This extension is licensed under the MIT License.
"#, self.config.version)
    }
}

/// Plugin factory for creating IDE plugins
pub struct PluginFactory;

impl PluginFactory {
    /// Create a VS Code plugin configuration
    #[allow(dead_code)]
pub fn create_vscode_config() -> PluginConfig {
        PluginConfig {
            ide_type: IdeType::VSCode,
            plugin_name: "augustium".to_string(),
            version: "1.0.0".to_string(),
            description: "Augustium language support for VS Code".to_string(),
            author: "Augustium Team".to_string(),
            features: vec![
                PluginFeature::SyntaxHighlighting,
                PluginFeature::AutoCompletion,
                PluginFeature::ErrorChecking,
                PluginFeature::Formatting,
                PluginFeature::Snippets,
                PluginFeature::BuildIntegration,
            ],
            dependencies: vec![
                "@types/vscode".to_string(),
                "typescript".to_string(),
            ],
            settings: HashMap::new(),
        }
    }
    
    /// Create an IntelliJ plugin configuration
    #[allow(dead_code)]
pub fn create_intellij_config() -> PluginConfig {
        PluginConfig {
            ide_type: IdeType::IntelliJ,
            plugin_name: "augustium-intellij".to_string(),
            version: "1.0.0".to_string(),
            description: "Augustium language support for IntelliJ IDEA".to_string(),
            author: "Augustium Team".to_string(),
            features: vec![
                PluginFeature::SyntaxHighlighting,
                PluginFeature::AutoCompletion,
                PluginFeature::ErrorChecking,
                PluginFeature::Navigation,
                PluginFeature::Refactoring,
            ],
            dependencies: vec![],
            settings: HashMap::new(),
        }
    }
    
    /// Create a Vim plugin configuration
    #[allow(dead_code)]
pub fn create_vim_config() -> PluginConfig {
        PluginConfig {
            ide_type: IdeType::Vim,
            plugin_name: "vim-augustium".to_string(),
            version: "1.0.0".to_string(),
            description: "Augustium language support for Vim".to_string(),
            author: "Augustium Team".to_string(),
            features: vec![
                PluginFeature::SyntaxHighlighting,
                PluginFeature::Formatting,
                PluginFeature::BuildIntegration,
            ],
            dependencies: vec![],
            settings: HashMap::new(),
        }
    }
    
    /// Generate plugins for all supported IDEs
    #[allow(dead_code)]
pub fn generate_all_plugins(output_dir: &Path) -> Result<(), CompilerError> {
        let configs = vec![
            Self::create_vscode_config(),
            Self::create_intellij_config(),
            Self::create_vim_config(),
        ];
        
        for config in configs {
            let plugin_dir = output_dir.join(format!("augustium-{}", config.ide_type.name().to_lowercase().replace(" ", "-")));
            let mut generator = IdePluginGenerator::new(config, plugin_dir);
            generator.generate()?;
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    
    #[test]
    fn test_ide_type_names() {
        assert_eq!(IdeType::VSCode.name(), "Visual Studio Code");
        assert_eq!(IdeType::IntelliJ.name(), "IntelliJ IDEA");
        assert_eq!(IdeType::Vim.name(), "Vim");
    }
    
    #[test]
    fn test_ide_config_dirs() {
        assert_eq!(IdeType::VSCode.config_dir(), ".vscode");
        assert_eq!(IdeType::IntelliJ.config_dir(), ".idea");
        assert_eq!(IdeType::Vim.config_dir(), ".vim");
    }
    
    #[test]
    fn test_plugin_config_creation() {
        let config = PluginFactory::create_vscode_config();
        assert_eq!(config.ide_type, IdeType::VSCode);
        assert_eq!(config.plugin_name, "augustium");
        assert!(!config.features.is_empty());
    }
    
    #[test]
    fn test_plugin_generator_creation() {
        let config = PluginFactory::create_vscode_config();
        let output_dir = env::temp_dir().join("test_plugin");
        let generator = IdePluginGenerator::new(config, output_dir);
        
        assert_eq!(generator.files.len(), 0); // No files generated yet
    }
    
    #[test]
    fn test_vscode_package_json_generation() {
        let config = PluginFactory::create_vscode_config();
        let output_dir = env::temp_dir().join("test_vscode");
        let generator = IdePluginGenerator::new(config, output_dir);
        
        let package_json = generator.create_vscode_package_json();
        assert!(package_json.contains("augustium"));
        assert!(package_json.contains("1.0.0"));
    }
    
    #[test]
    fn test_textmate_grammar_generation() {
        let config = PluginFactory::create_vscode_config();
        let output_dir = env::temp_dir().join("test_grammar");
        let generator = IdePluginGenerator::new(config, output_dir);
        
        let grammar = generator.create_textmate_grammar();
        assert!(grammar.contains("source.augustium"));
        assert!(grammar.contains("contract"));
        assert!(grammar.contains("function"));
    }
    
    #[test]
    fn test_vim_syntax_generation() {
        let config = PluginFactory::create_vim_config();
        let output_dir = env::temp_dir().join("test_vim");
        let generator = IdePluginGenerator::new(config, output_dir);
        
        let syntax = generator.create_vim_syntax();
        assert!(syntax.contains("syn keyword augKeyword"));
        assert!(syntax.contains("contract"));
        assert!(syntax.contains("function"));
    }
}