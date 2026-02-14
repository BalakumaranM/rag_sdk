# Git Hooks for RAG SDK

Automated quality checks that run before commits and pushes to ensure code quality and consistency.

## 📋 Overview

This project uses Git hooks to automatically enforce code quality standards:

- **pre-commit**: Runs before each commit (fast checks)
- **pre-push**: Runs before pushing to remote (comprehensive checks)
- **commit-msg**: Validates commit message format

## 🚀 Quick Start

### Installation

Run the setup script from your project root:

```bash
chmod +x setup-hooks.sh
./setup-hooks.sh
```

This will:
1. Copy hooks to `.git/hooks/`
2. Make them executable
3. Check for required dependencies

### Install Required Dependencies

```bash
# Using pip
pip install ruff mypy pytest pytest-cov bandit safety

# Using poetry
poetry add --group dev ruff mypy pytest pytest-cov bandit safety
```

## 🔍 Hook Details

### Pre-Commit Hook

Runs **before each commit**. Fast checks to catch issues early.

**Checks:**
1. ✅ **Ruff linting** - Code quality and style
2. ✅ **Ruff formatting** - Code formatting consistency
3. ✅ **MyPy type checking** - Static type validation
4. ✅ **Print statement detection** - Enforce logging instead of print()
5. ✅ **Debugger detection** - Find forgotten breakpoints/pdb
6. ✅ **Secret scanning** - Detect API keys, passwords, tokens
7. ✅ **File size check** - Warn about large files

**Example output:**
```
🔍 Running pre-commit checks...

Files to check:
rag_sdk/embeddings/openai.py
rag_sdk/vectorstore/pinecone.py

1. Running Ruff linter...
✓ Ruff linting passed

2. Checking code formatting...
✓ Code formatting is correct

3. Running MyPy type checker...
✓ Type checking passed

4. Checking for common issues...
✓ No print() statements found
✓ No debugger statements found

5. Checking for secrets...
✓ No secrets detected

6. Checking file sizes...
✓ No large files detected

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ All pre-commit checks passed!
```

### Pre-Push Hook

Runs **before pushing to remote**. Comprehensive checks including tests.

**Checks:**
1. ✅ **Comprehensive linting** - Full codebase lint
2. ✅ **Type checking** - Strict mode on entire project
3. ✅ **Unit tests** - All unit tests must pass
4. ✅ **Integration tests** - If present, must pass
5. ✅ **Test coverage** - Minimum 80% coverage required
6. ✅ **Security scan** - Bandit security checks
7. ✅ **Dependency vulnerabilities** - Safety check
8. ✅ **Import verification** - Package imports correctly
9. ✅ **Documentation build** - Sphinx docs build successfully
10. ⚠️ **Main branch protection** - Confirm before pushing to main/master

**Example output:**
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚀 Running pre-push checks...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Branch: feature/add-embeddings

1. Running comprehensive linting...
✓ Linting passed

2. Running type checker on entire codebase...
✓ Type checking passed

3. Running unit tests...
✓ Unit tests passed

4. Running integration tests...
✓ Integration tests passed

5. Checking test coverage...
✓ Test coverage: 87% (minimum: 80%)

6. Running security checks...
✓ Security scan passed

7. Checking dependencies for vulnerabilities...
✓ No known vulnerabilities in dependencies

8. Checking for uncommitted changes...
✓ No uncommitted changes

9. Verifying package imports...
✓ Package imports successfully

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ All pre-push checks passed!
Safe to push to remote
```

### Commit Message Hook

Enforces **Conventional Commits** format for consistent commit history.

**Format:**
```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Valid types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style/formatting
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Test changes
- `build`: Build system changes
- `ci`: CI/CD changes
- `chore`: Other changes

**Examples:**
```bash
# Good ✅
git commit -m "feat(embeddings): add Cohere embedding provider"
git commit -m "fix(vectorstore): handle connection timeout properly"
git commit -m "docs(readme): update installation instructions"
git commit -m "test(retrieval): add hybrid search unit tests"

# Bad ❌
git commit -m "added stuff"
git commit -m "Fixed bug"
git commit -m "WIP"
```

## 🛠️ Configuration

### Customizing Checks

Edit the hook files in your project root (before running setup):

**Skip specific checks:**
```bash
# In pre-commit, comment out a check:
# Check 3: MyPy type checking
# echo -e "${YELLOW}3. Running MyPy type checker...${NC}"
# if command -v mypy &> /dev/null; then
#     ...
# fi
```

**Adjust thresholds:**
```bash
# In pre-push, change coverage threshold:
--cov-fail-under=80  # Change to desired percentage
```

### Project-Specific Configuration

Create `.ruff.toml`:
```toml
line-length = 100
target-version = "py39"

[lint]
select = ["E", "F", "W", "I", "N", "UP", "B", "A", "C4", "DTZ", "T10", "EM", "ISC", "ICN", "G", "PIE", "T20", "PYI", "PT", "Q", "RSE", "RET", "SLF", "SIM", "TID", "TCH", "ARG", "PTH", "ERA", "PD", "PL", "TRY", "NPY", "RUF"]
ignore = ["E501"]  # Line too long (handled by formatter)
```

Create `mypy.ini`:
```ini
[mypy]
python_version = 3.9
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
disallow_incomplete_defs = True
check_untyped_defs = True
no_implicit_optional = True
warn_redundant_casts = True
warn_unused_ignores = True
warn_no_return = True
strict_equality = True
```

Create `pytest.ini`:
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --strict-markers
    --tb=short
    --cov-report=term-missing
```

## 🔄 Bypassing Hooks

**Not recommended**, but sometimes necessary:

```bash
# Skip pre-commit hook
git commit --no-verify -m "feat: emergency hotfix"

# Skip pre-push hook
git push --no-verify
```

**When to bypass:**
- Emergency hotfixes (fix ASAP, clean up later)
- Work-in-progress commits on feature branch
- CI/CD is running the same checks anyway

**Never bypass on:**
- Commits to main/master branch
- Production releases
- Public commits

## 📊 Continuous Integration

These hooks complement CI/CD, not replace it:

**Local (Git Hooks):**
- ✅ Fast feedback
- ✅ Catches issues before commit/push
- ✅ Saves CI/CD resources
- ❌ Can be bypassed

**CI/CD (GitHub Actions, etc.):**
- ✅ Cannot be bypassed
- ✅ Runs on all PRs
- ✅ Official record
- ❌ Slower feedback loop

**Recommended workflow:**
1. Local hooks catch most issues immediately
2. CI/CD provides final verification
3. Both use same tools (ruff, mypy, pytest)

### Example GitHub Actions

```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -e ".[dev]"
    
    - name: Run linting
      run: ruff check .
    
    - name: Run type checking
      run: mypy rag_sdk/
    
    - name: Run tests with coverage
      run: pytest --cov=rag_sdk --cov-report=xml
    
    - name: Security check
      run: bandit -r rag_sdk/
```

## 🐛 Troubleshooting

### Hooks not running

```bash
# Check if hooks are installed
ls -la .git/hooks/

# Reinstall hooks
./setup-hooks.sh

# Check if executable
chmod +x .git/hooks/pre-commit
chmod +x .git/hooks/pre-push
chmod +x .git/hooks/commit-msg
```

### Dependency issues

```bash
# Verify installations
which ruff
which mypy
which pytest

# Reinstall dependencies
pip install --upgrade ruff mypy pytest pytest-cov bandit
```

### Path issues

Hooks run in the repository root. If tools aren't found:

```bash
# Check your PATH
echo $PATH

# Use absolute paths in hooks
/usr/local/bin/ruff check .
```

### Hook failures

```bash
# Run manually to debug
.git/hooks/pre-commit
.git/hooks/pre-push

# Check specific tool
ruff check rag_sdk/
mypy rag_sdk/
pytest tests/
```

## 🎯 Best Practices

### For Developers

1. **Install hooks immediately** after cloning the repository
2. **Don't bypass hooks** unless absolutely necessary
3. **Fix issues locally** before pushing
4. **Keep commits small** - easier to pass pre-commit checks
5. **Write tests** as you code - not after
6. **Use descriptive commit messages** following conventions

### For Teams

1. **Document hook requirements** in main README
2. **Include hook setup in onboarding** process
3. **Keep hooks fast** - slow hooks get bypassed
4. **Align with CI/CD** - same checks in both places
5. **Review hook failures** in standup/retrospectives
6. **Update hooks** when adding new tools

## 📚 Additional Resources

- [Conventional Commits](https://www.conventionalcommits.org/)
- [Git Hooks Documentation](https://git-scm.com/docs/githooks)
- [Ruff Linter](https://docs.astral.sh/ruff/)
- [MyPy](https://mypy.readthedocs.io/)
- [Pytest](https://docs.pytest.org/)
- [Bandit Security](https://bandit.readthedocs.io/)

## 🤝 Contributing

To modify hooks:

1. Edit hook files in project root
2. Test changes manually
3. Run `./setup-hooks.sh` to update
4. Commit changes to version control
5. Team members run setup script to get updates

## 📝 License

Same as the RAG SDK project.
