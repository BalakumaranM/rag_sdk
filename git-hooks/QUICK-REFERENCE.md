# Git Hooks Quick Reference

## 🚀 Setup
```bash
./setup-hooks.sh
```

## 📝 Commit Message Format
```
<type>(<scope>): <description>

feat(embeddings): add Cohere support
fix(vectorstore): handle timeout errors
docs(readme): update installation guide
test(retrieval): add hybrid search tests
```

## 🎯 Valid Types
| Type | Description | Example |
|------|-------------|---------|
| `feat` | New feature | `feat(llm): add Claude support` |
| `fix` | Bug fix | `fix(config): validate API keys` |
| `docs` | Documentation | `docs(api): update docstrings` |
| `style` | Formatting | `style: format with ruff` |
| `refactor` | Code restructure | `refactor(core): simplify pipeline` |
| `perf` | Performance | `perf(search): optimize indexing` |
| `test` | Tests | `test(unit): add coverage for X` |
| `build` | Build system | `build: update dependencies` |
| `ci` | CI/CD | `ci: add GitHub Actions` |
| `chore` | Maintenance | `chore: update .gitignore` |

## ⚡ Pre-Commit Checks
Runs on: `git commit`
- ✅ Ruff linting
- ✅ Code formatting
- ✅ Type checking (MyPy)
- ✅ No print() statements
- ✅ No debugger statements
- ✅ Secret detection
- ✅ File size check

## 🔒 Pre-Push Checks
Runs on: `git push`
- ✅ Full linting
- ✅ Strict type checking
- ✅ Unit tests
- ✅ Integration tests
- ✅ 80%+ test coverage
- ✅ Security scan (Bandit)
- ✅ Vulnerability check
- ✅ Package imports
- ✅ Documentation build
- ⚠️ Main branch protection

## 🛠️ Bypass Hooks (Emergency Only)
```bash
git commit --no-verify -m "fix: hotfix"
git push --no-verify
```

## 📦 Required Dependencies
```bash
pip install ruff mypy pytest pytest-cov bandit safety
```

## 🐛 Troubleshooting
```bash
# Reinstall hooks
./setup-hooks.sh

# Make executable
chmod +x .git/hooks/*

# Test manually
.git/hooks/pre-commit
.git/hooks/pre-push
```

## ✨ Tips
1. Install hooks immediately after clone
2. Keep commits small and focused
3. Write tests as you code
4. Fix issues before pushing
5. Never bypass on main branch
6. Use conventional commit messages
