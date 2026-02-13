# Git Workflow Rules - Part 1: Core Guidelines

## 🎯 Core Principle
**Commit early, commit often.** After every meaningful change or completion of a logical unit of work, the agent MUST commit and push the changes to the repository.

---

## 📋 When to Commit

### ALWAYS Commit After:
1. ✅ Creating a new file or module
2. ✅ Implementing a complete function or class
3. ✅ Adding or modifying tests
4. ✅ Fixing a bug
5. ✅ Refactoring code
6. ✅ Updating documentation
7. ✅ Adding dependencies
8. ✅ Completing a feature or sub-feature
9. ✅ Making configuration changes
10. ✅ Any change that makes the code work better

### Commit Frequency Guidelines:
- **Minimum**: After every file creation or major change
- **Recommended**: Every 15-30 minutes of work
- **Maximum**: Never go more than 1 hour without committing
- **Rule of thumb**: If you've written 50+ lines of code, commit it

### DO NOT Commit:
- ❌ Broken or non-functional code (unless marked as WIP)
- ❌ Code with syntax errors
- ❌ Secrets, API keys, or sensitive data
- ❌ Large binary files without discussion
- ❌ Generated files (unless necessary)

---

## 📝 Commit Message Format

### STRICT REQUIREMENT: Conventional Commits

Every commit MUST follow this format:
```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Valid Types:
| Type | When to Use | Example |
|------|-------------|---------|
| `feat` | New feature added | `feat(embeddings): add OpenAI provider` |
| `fix` | Bug fix | `fix(vectorstore): handle connection timeout` |
| `docs` | Documentation only | `docs(readme): update installation steps` |
| `style` | Code formatting (no logic change) | `style(core): format with ruff` |
| `refactor` | Code restructure (no behavior change) | `refactor(retrieval): simplify search logic` |
| `perf` | Performance improvement | `perf(embeddings): add batch processing` |
| `test` | Adding/updating tests | `test(vectorstore): add unit tests for FAISS` |
| `build` | Build system/dependencies | `build: add pytest-asyncio dependency` |
| `ci` | CI/CD changes | `ci: add GitHub Actions workflow` |
| `chore` | Maintenance tasks | `chore: update .gitignore` |

### Scope Examples:
- `embeddings` - Embedding module
- `vectorstore` - Vector store module
- `llm` - LLM generation module
- `retrieval` - Retrieval module
- `config` - Configuration system
- `core` - Core abstractions
- `docs` - Documentation
- `tests` - Test files
- `security` - Security features
- `api` - Public API

### Commit Message Examples:

**Good Examples ✅**
```
feat(embeddings): implement OpenAI embedding provider
fix(vectorstore): add retry logic for connection failures
docs(api): add docstrings to EmbeddingProvider class
test(retrieval): add unit tests for hybrid search
refactor(config): use Pydantic for validation
perf(embeddings): batch API calls to reduce latency
build: add pytest and mypy to dev dependencies
```

**Bad Examples ❌**
```
"added stuff"
"Update file.py"
"WIP"
"fix"
"changes"
"test commit"
```

---

## 🔄 Git Workflow Commands

### Step 1: Check Status
```bash
git status
```

### Step 2: Add Files
**Add specific files (preferred):**
```bash
git add rag_sdk/embeddings/openai.py
git add tests/unit/test_embeddings.py
```

**Never add:**
```bash
git add .env              # Environment variables
git add *.pyc             # Python cache
git add __pycache__/      # Python cache dirs
```

### Step 3: Commit
```bash
git commit -m "feat(embeddings): add OpenAI provider"
```

### Step 4: Push
```bash
git push
```

---

## 🤖 Agent Workflow Protocol

### After Every Code Change:

1. **Verify code works**
   ```bash
   python -m py_compile rag_sdk/embeddings/openai.py
   pytest tests/unit/test_embeddings.py
   ```

2. **Check what changed**
   ```bash
   git status
   git diff
   ```

3. **Stage files**
   ```bash
   git add <files>
   ```

4. **Commit with proper message**
   ```bash
   git commit -m "<type>(<scope>): <description>"
   ```

5. **Push to remote**
   ```bash
   git push
   ```

6. **Confirm to user**
   ```
   ✅ Changes committed and pushed:
   - Added OpenAI embedding provider
   - Commit: feat(embeddings): implement OpenAI provider
   - Branch: feature/add-embeddings
   ```

---

## 🎨 Agent Response Template

After making changes, the agent should respond:

```
I've implemented [description of what was done].

📝 Files changed:
- rag_sdk/embeddings/openai.py (new)
- tests/unit/test_openai_embeddings.py (new)
- docs/embeddings.md (updated)

🔄 Committing changes...
✅ Committed: feat(embeddings): implement OpenAI embedding provider
✅ Pushed to: origin/feature/add-embeddings

Next steps:
1. Review the implementation
2. Test with: pytest tests/unit/test_openai_embeddings.py
3. Ready to move to the next component
```

---

## 🌳 Branch Management

### Branch Naming Convention:
```
<type>/<short-description>

Examples:
feature/add-embeddings
fix/connection-timeout
docs/api-documentation
refactor/config-system
```

### Creating Branches:
```bash
git checkout -b feature/add-cohere-embeddings
# or
git switch -c feature/add-cohere-embeddings
```

---

## ⚠️ Error Handling

### If Git Hooks Fail:

**Agent should:**
1. Read the error message
2. Fix the issues (linting, formatting, etc.)
3. Re-stage the files
4. Commit again

**Example Response:**
```
⚠️ Pre-commit hook detected linting issues. Fixing...

🔧 Running: ruff format rag_sdk/embeddings/openai.py
✅ Fixed formatting issues

🔄 Re-committing...
✅ Committed: feat(embeddings): implement OpenAI provider
✅ Pushed successfully
```

### If Push Fails:

```bash
git pull --rebase origin main
git push
```

**Agent Response:**
```
⚠️ Push failed - remote has new changes

🔄 Pulling latest changes...
✅ Rebased successfully
✅ Pushed successfully
```

### If Conflicts Occur:

**Agent should:**
- Alert the user immediately
- Explain the conflict
- Ask for guidance
- DO NOT auto-resolve complex conflicts

---

## 🛡️ Safety Checks Before Commit

### Pre-Commit Checklist:
1. ✅ Code has no syntax errors
2. ✅ Tests pass (if applicable)
3. ✅ No secrets or API keys in code
4. ✅ No debugging statements (print, breakpoint, pdb)
5. ✅ Files are properly formatted
6. ✅ Imports are working
7. ✅ Type hints are added
8. ✅ Docstrings are present

---

## 🚀 Pushing Strategy

### When to Push:

**ALWAYS push after:**
1. Completing a feature or component
2. Fixing a bug
3. End of work session
4. Before asking user for review
5. After every 3-5 commits

**Push frequency:**
- **Minimum**: Every feature/fix completion
- **Recommended**: Every 30 minutes to 1 hour
- **Maximum**: End of each interaction

---

## 🎯 Core Reminders

1. **Commit early, commit often** - Don't accumulate changes
2. **Write good messages** - Follow conventional commits
3. **Push regularly** - Backup work to remote
4. **Keep commits atomic** - One logical change per commit
5. **Check before committing** - No secrets or errors
6. **Inform the user** - Tell them what was committed

### Never:
1. ❌ Commit secrets or API keys
2. ❌ Force push to main/master
3. ❌ Make huge commits (1000+ lines)
4. ❌ Use vague commit messages
5. ❌ Leave code uncommitted overnight
6. ❌ Ignore git hook failures

---

**Continue to Part 2 for Advanced Workflows & Examples →**
