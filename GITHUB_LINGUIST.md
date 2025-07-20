# GitHub Linguist Integration for Augustium

This document explains how to get the Augustium programming language recognized by GitHub's language detection system.

## What We've Added

### 1. `.gitattributes` File
The `.gitattributes` file tells GitHub how to handle Augustium files:
```
*.aug linguist-language=Augustium
```

### 2. Language Definition Files
- `.github/linguist.yml` - Local configuration for this repository
- `linguist-definition.yml` - Reference definition for official submission

## Getting Augustium Officially Recognized

To get Augustium added to GitHub Linguist's official language database:

### Step 1: Meet the Requirements
- [ ] Language must be in use (✅ - Augustium has examples and documentation)
- [ ] Language must have a unique file extension (✅ - `.aug`)
- [ ] Language must have a grammar/syntax definition
- [ ] Language should have multiple repositories using it

### Step 2: Create a Pull Request to GitHub Linguist

1. Fork the [GitHub Linguist repository](https://github.com/github/linguist)

2. Add Augustium to `lib/linguist/languages.yml`:
```yaml
Augustium:
  type: programming
  color: "#ff6b35"
  extensions:
  - ".aug"
  tm_scope: source.augustium
  ace_mode: text
  language_id: 1001
  aliases:
  - augustium
  - aug
```

3. Add detection patterns to `lib/linguist/heuristics.yml`:
```yaml
".aug":
- pattern: '^\s*contract\s+\w+\s*\{'
  language: Augustium
- pattern: '^\s*use\s+std::'
  language: Augustium
- pattern: '\bpub\s+fn\s+\w+\s*\('
  language: Augustium
```

4. Add sample files to `samples/Augustium/`:
   - Copy `examples/hello_world.aug`
   - Copy `examples/simple-token.aug`
   - Ensure files are representative of the language

5. Run tests:
```bash
bundle exec rake test
```

6. Submit the pull request with:
   - Clear description of the Augustium language
   - Links to documentation and examples
   - Evidence of the language being used in multiple projects

### Step 3: TextMate Grammar (Optional but Recommended)

For better syntax highlighting, create a TextMate grammar file:
- File: `grammars/source.augustium.json`
- Define syntax patterns for keywords, strings, comments, etc.
- Reference the existing IDE plugin work in `src/ide_plugins.rs`

## Current Status

✅ **Local Configuration Complete**
- `.gitattributes` file created
- Local linguist configuration added
- Language definition documented

⏳ **Next Steps**
- Create TextMate grammar for better syntax highlighting
- Submit pull request to GitHub Linguist repository
- Encourage community adoption to meet usage requirements

## Testing Locally

To test language detection locally:

1. Install GitHub Linguist:
```bash
gem install github-linguist
```

2. Run detection on your repository:
```bash
github-linguist
```

3. Check specific files:
```bash
github-linguist examples/hello_world.aug
```

## Resources

- [GitHub Linguist Documentation](https://github.com/github/linguist/blob/master/CONTRIBUTING.md)
- [Language Addition Guidelines](https://github.com/github/linguist/blob/master/CONTRIBUTING.md#adding-a-language)
- [TextMate Grammar Guide](https://macromates.com/manual/en/language_grammars)
- [Augustium Language Examples](./examples/)

## Community Help

To increase chances of acceptance:
1. Create more Augustium projects on GitHub
2. Encourage others to use the `.aug` extension
3. Build a community around the language
4. Document the language specification thoroughly

Once we have sufficient community adoption, we can submit the official pull request to GitHub Linguist!