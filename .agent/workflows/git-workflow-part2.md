# Git Workflow Rules - Part 2: Advanced Workflows

## 📊 Commit Best Practices

### Atomic Commits
Each commit should represent ONE logical change:

**Good (atomic):**
```bash
git commit -m "feat(embeddings): add OpenAI provider"
git commit -m "test(embeddings): add tests for OpenAI provider"
git commit -m "docs(embeddings): document OpenAI integration"
```

**Bad (not atomic):**
```bash
git commit -m "feat: add OpenAI, fix bugs, update docs, refactor config"
```

### Meaningful Messages
```
# Good ✅
feat(vectorstore): implement connection pooling for Pinecone
fix(retrieval): handle empty query results gracefully
refactor(config): migrate from JSON to YAML format

# Bad ❌
"update"
"fix stuff"
"made changes"
```

### Commit Size Guidelines:
- **Ideal**: 50-200 lines changed
- **Acceptable**: Up to 500 lines for new files
- **Too large**: 1000+ lines (consider breaking down)

---

## 🔍 Special Scenarios

### Scenario 1: Working on Multiple Files

```bash
# Complete Feature A
git add rag_sdk/embeddings/openai.py tests/unit/test_openai.py
git commit -m "feat(embeddings): add OpenAI provider with tests"
git push

# Complete Feature B
git add rag_sdk/embeddings/cohere.py tests/unit/test_cohere.py
git commit -m "feat(embeddings): add Cohere provider with tests"
git push
```

### Scenario 2: Found Bug While Coding

```bash
# Stop current work, fix bug first
git add rag_sdk/config/settings.py
git commit -m "fix(config): validate API key format properly"
git push

# Continue with original work
git add rag_sdk/embeddings/openai.py
git commit -m "feat(embeddings): add OpenAI provider"
git push
```

### Scenario 3: Experimental Code

```bash
# Create experimental branch
git checkout -b experiment/new-retrieval-algorithm

# Make changes
git add rag_sdk/retrieval/experimental.py
git commit -m "feat(retrieval): experiment with new ranking algorithm"
git push origin experiment/new-retrieval-algorithm

# Inform user it's experimental
```

### Scenario 4: Large Feature (Break It Down)

For large features, commit incrementally:

```bash
# Step 1: Add interface
git add rag_sdk/embeddings/base.py
git commit -m "feat(embeddings): add EmbeddingProvider interface"
git push

# Step 2: Implement OpenAI
git add rag_sdk/embeddings/openai.py
git commit -m "feat(embeddings): implement OpenAI provider"
git push

# Step 3: Add factory
git add rag_sdk/embeddings/factory.py
git commit -m "feat(embeddings): add provider factory pattern"
git push

# Step 4: Add tests
git add tests/unit/test_embeddings.py
git commit -m "test(embeddings): add comprehensive test suite"
git push

# Step 5: Add documentation
git add docs/embeddings.md
git commit -m "docs(embeddings): document embedding providers"
git push
```

---

## 🎓 Complete Workflow Example

### Example: Adding Cohere Embeddings

```
USER: "Add support for Cohere embeddings"

AGENT: 
[Analyzes requirements, plans implementation]

I'll implement Cohere embeddings support. Let me break this into steps:
1. Create Cohere provider class
2. Add unit tests
3. Update factory
4. Add documentation

Step 1: Creating Cohere provider class...

[Writes code in rag_sdk/embeddings/cohere.py]

📝 Files created:
- rag_sdk/embeddings/cohere.py (145 lines)

🔄 Committing Step 1...

$ git add rag_sdk/embeddings/cohere.py
$ git commit -m "feat(embeddings): implement Cohere embedding provider"
[feature/add-embeddings abc1234] feat(embeddings): implement Cohere embedding provider
 1 file changed, 145 insertions(+)
 create mode 100644 rag_sdk/embeddings/cohere.py

$ git push origin feature/add-embeddings
✅ Pushed to origin/feature/add-embeddings

Step 2: Adding unit tests...

[Writes tests in tests/unit/test_cohere_embeddings.py]

📝 Files created:
- tests/unit/test_cohere_embeddings.py (87 lines)

🔄 Committing Step 2...

$ git add tests/unit/test_cohere_embeddings.py
$ git commit -m "test(embeddings): add unit tests for Cohere provider"
[feature/add-embeddings def5678] test(embeddings): add unit tests for Cohere provider
 1 file changed, 87 insertions(+)
 create mode 100644 tests/unit/test_cohere_embeddings.py

$ git push origin feature/add-embeddings
✅ Pushed to origin/feature/add-embeddings

Step 3: Updating factory...

[Updates rag_sdk/embeddings/factory.py]

📝 Files modified:
- rag_sdk/embeddings/factory.py

🔄 Committing Step 3...

$ git add rag_sdk/embeddings/factory.py
$ git commit -m "feat(embeddings): add Cohere to provider factory"
[feature/add-embeddings ghi9012] feat(embeddings): add Cohere to provider factory
 1 file changed, 8 insertions(+), 2 deletions(-)

$ git push origin feature/add-embeddings
✅ Pushed to origin/feature/add-embeddings

Step 4: Adding documentation...

[Updates docs/embeddings.md]

📝 Files modified:
- docs/embeddings.md

🔄 Committing Step 4...

$ git add docs/embeddings.md
$ git commit -m "docs(embeddings): document Cohere integration"
[feature/add-embeddings jkl3456] docs(embeddings): document Cohere integration
 1 file changed, 45 insertions(+), 0 deletions(-)

$ git push origin feature/add-embeddings
✅ Pushed to origin/feature/add-embeddings

✅ COMPLETE! Cohere embeddings support fully implemented.

Summary of commits:
1. feat(embeddings): implement Cohere embedding provider
2. test(embeddings): add unit tests for Cohere provider
3. feat(embeddings): add Cohere to provider factory
4. docs(embeddings): document Cohere integration

All changes have been pushed to: origin/feature/add-embeddings

Would you like me to:
1. Add integration tests
2. Add example usage
3. Move to the next provider
```

---

## 📈 Agent Success Metrics

The agent is doing well if:
- ✅ Commits happen every 15-30 minutes
- ✅ Each commit message follows conventions
- ✅ Commits are atomic (single purpose)
- ✅ Code is pushed after each commit or set of related commits
- ✅ No large uncommitted changes
- ✅ Git hooks pass consistently
- ✅ Branch names are meaningful
- ✅ User can track progress via commits

The agent needs improvement if:
- ❌ Hours pass without commits
- ❌ Commit messages are vague
- ❌ Giant commits with multiple unrelated changes
- ❌ Code sitting uncommitted for long periods
- ❌ Frequent git hook failures
- ❌ Force pushing to shared branches

---

## 🎯 Detailed Response Patterns

### Pattern 1: Single File Change
```
I've [action taken].

📝 File: rag_sdk/embeddings/openai.py
📊 Changes: 120 lines added

🔄 Git workflow:
$ git add rag_sdk/embeddings/openai.py
$ git commit -m "feat(embeddings): add OpenAI provider"
✅ Committed (sha: abc1234)
$ git push
✅ Pushed to origin/feature/add-embeddings

The implementation includes:
- Async API client
- Batch processing support
- Error handling with retries
- Type hints and docstrings

Ready for next step!
```

### Pattern 2: Multiple Related Files
```
I've implemented [feature] with tests and documentation.

📝 Files:
- rag_sdk/vectorstore/pinecone.py (new, 180 lines)
- tests/unit/test_pinecone.py (new, 95 lines)
- docs/vectorstore.md (updated, +50 lines)

🔄 Git workflow:
$ git add rag_sdk/vectorstore/pinecone.py tests/unit/test_pinecone.py
$ git commit -m "feat(vectorstore): implement Pinecone adapter with tests"
✅ Committed (sha: abc1234)

$ git add docs/vectorstore.md
$ git commit -m "docs(vectorstore): document Pinecone integration"
✅ Committed (sha: def5678)

$ git push
✅ All changes pushed to origin/feature/add-vectorstores

Next: Would you like to add Weaviate or Qdrant next?
```

### Pattern 3: Bug Fix During Development
```
⚠️ Found a bug in the config validation logic while implementing [feature].

Fixing the bug first...

📝 Fixed: rag_sdk/config/validator.py
🐛 Issue: API key validation was case-sensitive

🔄 Git workflow:
$ git add rag_sdk/config/validator.py
$ git commit -m "fix(config): make API key validation case-insensitive"
✅ Committed and pushed (sha: xyz9876)

Now continuing with [feature]...
```

### Pattern 4: Incremental Feature Development
```
Working on [large feature]...

Progress update:

Step 1/4: Interface definition
✅ Committed: feat(retrieval): add Retriever base class
✅ Pushed (sha: aaa1111)

Step 2/4: Dense retrieval implementation  
✅ Committed: feat(retrieval): implement dense vector search
✅ Pushed (sha: bbb2222)

Step 3/4: Currently working on hybrid search...
[Will commit when complete]
```

---

## 🚦 Decision Tree for Commits

```
Has meaningful code been written?
├─ YES: Is it a complete logical unit?
│  ├─ YES: Are there syntax errors?
│  │  ├─ YES: Fix errors first, then commit
│  │  └─ NO: Is it properly formatted?
│  │     ├─ YES: COMMIT NOW ✅
│  │     └─ NO: Format, then commit
│  └─ NO: Continue working until unit is complete
└─ NO: Keep coding
```

---

## 🔐 Security Reminders

### Before Every Commit, Check:

```bash
# 1. Search for API keys
grep -r "sk-" rag_sdk/ tests/
grep -r "api_key.*=" rag_sdk/ tests/

# 2. Search for secrets
grep -r "password.*=" rag_sdk/
grep -r "secret.*=" rag_sdk/

# 3. Check .env is gitignored
cat .gitignore | grep ".env"

# 4. Review staged changes
git diff --cached
```

### If Secrets Found:
```
❌ STOP! Secrets detected in staged files!

🔍 Found in: rag_sdk/config/settings.py
Line 15: api_key = "sk-abc123..."

Action required:
1. Remove hardcoded secret
2. Use environment variable instead
3. Update .env.example with placeholder
4. Recommit without secret

Would you like me to fix this automatically?
```

---

## 📚 Quick Reference for Agent

### Commit After:
- ✅ New file created (complete)
- ✅ Function/class implemented
- ✅ Tests added/updated
- ✅ Bug fixed
- ✅ Documentation updated
- ✅ Every 30-60 minutes

### Commit Message Format:
```
<type>(<scope>): <description>

Types: feat, fix, docs, style, refactor, perf, test, build, ci, chore
Scopes: embeddings, vectorstore, llm, retrieval, config, core, docs, tests
```

### Push After:
- ✅ Every commit (or every 3-5 commits)
- ✅ Feature complete
- ✅ Before asking for user review
- ✅ End of work session

### Error Handling:
- 🔧 Hook fails → Fix issue → Recommit
- 🔄 Push fails → Pull with rebase → Push
- ⚠️ Conflict → Alert user → Don't auto-resolve

---

## 🎉 Final Checklist

Before ending each interaction, ensure:
- [ ] All code changes are committed
- [ ] All commits follow conventional format
- [ ] All commits are pushed to remote
- [ ] User is informed of what was committed
- [ ] Next steps are clear
- [ ] No uncommitted changes remain

**Remember: The best commit is the one that happens NOW! Don't wait, commit and push! 🚀**

---

**← See Part 1 for Core Guidelines and Commit Format**
