# Git Hooks Installation Guide for RAG SDK

## 📁 Files Included

```
git-hooks/
├── pre-commit           # Pre-commit hook script
├── pre-push            # Pre-push hook script
├── commit-msg          # Commit message validation hook
├── setup-hooks.sh      # Installation script
├── GIT-HOOKS-README.md # Full documentation
└── QUICK-REFERENCE.md  # Quick reference card
```

## 🚀 Installation Steps

### 1. Copy Files to Your Project

Copy all files to your RAG SDK project root:

```bash
cd /path/to/rag-sdk
mkdir -p git-hooks
cp /path/to/downloaded/files/* git-hooks/
```

### 2. Make Setup Script Executable

```bash
chmod +x git-hooks/setup-hooks.sh
```

### 3. Run Setup Script

```bash
cd /path/to/rag-sdk  # Must be in project root
./git-hooks/setup-hooks.sh
```

This will:
- Copy hooks to `.git/hooks/`
- Make them executable
- Check for dependencies
- Report installation status

### 4. Install Dependencies

```bash
# Using pip
pip install ruff mypy pytest pytest-cov bandit safety

# Using poetry (recommended)
poetry add --group dev ruff mypy pytest pytest-cov bandit safety
```

### 5. Verify Installation

```bash
# Check hooks are installed
ls -la .git/hooks/

# Test pre-commit manually
.git/hooks/pre-commit

# Test with a dummy commit
git add -A
git commit -m "test: verify hooks"
```

## ✅ What Happens After Installation

### On Every Commit (`git commit`)
The **pre-commit** hook runs automatically:
- Lints your code with Ruff
- Checks formatting
- Validates types with MyPy
- Scans for print() statements
- Detects secrets/API keys
- Takes ~5-10 seconds

### On Every Push (`git push`)
The **pre-push** hook runs automatically:
- Runs all pre-commit checks
- Executes full test suite
- Checks test coverage (min 80%)
- Runs security scan
- Validates imports
- Takes ~30-60 seconds

### On Every Commit Message
The **commit-msg** hook validates format:
```
✅ feat(embeddings): add OpenAI support
✅ fix(vectorstore): handle timeouts
❌ "fixed stuff"
❌ "WIP"
```

## 🎯 Quick Start After Installation

### Making Your First Commit

```bash
# 1. Make changes
vim rag_sdk/embeddings/openai.py

# 2. Add files
git add rag_sdk/embeddings/openai.py

# 3. Commit (hooks run automatically)
git commit -m "feat(embeddings): add OpenAI embedding provider"

# Output:
# 🔍 Running pre-commit checks...
# ✓ Ruff linting passed
# ✓ Code formatting is correct
# ✓ Type checking passed
# ...
# ✓ All pre-commit checks passed!
```

### Pushing to Remote

```bash
# Push (comprehensive checks run)
git push origin feature/add-embeddings

# Output:
# 🚀 Running pre-push checks...
# ✓ Linting passed
# ✓ Type checking passed
# ✓ Unit tests passed (127 passed)
# ✓ Test coverage: 87%
# ...
# ✓ All pre-push checks passed!
# Safe to push to remote
```

## 🔧 Optional: Project Configuration

### Create `.ruff.toml`

```toml
line-length = 100
target-version = "py39"

[lint]
select = ["E", "F", "W", "I", "N", "UP", "B", "C4", "T20", "RET", "SIM", "ARG", "PTH", "PL", "RUF"]
ignore = ["E501"]
```

### Create `mypy.ini`

```ini
[mypy]
python_version = 3.9
strict = True
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
```

### Create `pytest.ini`

```ini
[pytest]
testpaths = tests
addopts = -v --tb=short --cov-report=term-missing
```

### Update `pyproject.toml`

```toml
[tool.poetry.group.dev.dependencies]
ruff = "^0.1.0"
mypy = "^1.7.0"
pytest = "^7.4.0"
pytest-cov = "^4.1.0"
pytest-asyncio = "^0.21.0"
bandit = "^1.7.5"
safety = "^2.3.5"
```

## 🐛 Troubleshooting

### Issue: Hooks not running

**Solution:**
```bash
# Reinstall
./git-hooks/setup-hooks.sh

# Verify installation
ls -la .git/hooks/pre-commit
ls -la .git/hooks/pre-push
ls -la .git/hooks/commit-msg
```

### Issue: "command not found: ruff"

**Solution:**
```bash
# Install in current environment
pip install ruff

# Or activate virtual environment first
source venv/bin/activate  # or: poetry shell
pip install ruff
```

### Issue: Tests failing in pre-push

**Solution:**
```bash
# Run tests manually to see full output
pytest tests/unit/ -v

# Fix failing tests before pushing
```

### Issue: Commit message rejected

**Solution:**
Use conventional commit format:
```bash
# Bad
git commit -m "fixed bug"

# Good
git commit -m "fix(vectorstore): handle connection errors"
```

## 🎓 Team Setup

### For New Team Members

Add to your project's main README.md:

```markdown
## Development Setup

1. Clone the repository
2. Install dependencies: `poetry install`
3. **Install Git hooks: `./git-hooks/setup-hooks.sh`** ← Important!
4. Verify setup: `git commit --allow-empty -m "test: verify hooks"`
```

### For Continuous Integration

Ensure CI runs same checks:

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - run: pip install -e ".[dev]"
      - run: ruff check .
      - run: mypy rag_sdk/
      - run: pytest --cov=rag_sdk --cov-fail-under=80
      - run: bandit -r rag_sdk/
```

## 📚 Next Steps

1. ✅ Install hooks (completed above)
2. 📖 Read `GIT-HOOKS-README.md` for full documentation
3. 📋 Print `QUICK-REFERENCE.md` for your desk
4. 🔧 Customize hooks for your needs
5. 👥 Share with team members
6. 🔄 Set up CI/CD with same checks

## 🎉 You're All Set!

Your Git hooks are now protecting code quality automatically. Every commit and push will be validated before it enters your repository.

**Happy coding! 🚀**
