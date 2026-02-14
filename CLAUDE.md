# Project Guidelines for Claude Code

## Git Commit Conventions

This project enforces **Conventional Commits** via git hooks. Every commit message MUST follow:

```
<type>(<scope>): <description>
```

**Valid types:** `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`

**Scope** is optional but must be lowercase alphanumeric with hyphens (e.g., `retrieval`, `config`, `graph-rag`).

**Examples:**
```
feat(retrieval): add Graph RAG retriever
fix(config): handle missing API key gracefully
test(generation): add CoVe unit tests
refactor(core): extract init methods
docs(readme): update installation instructions
```

**Rules:**
- Subject line max 100 characters
- Blank line between subject and body
- NEVER use `--no-verify` to skip git hooks
- NEVER force push

## Git Hooks (Active)

Three git hooks are installed in `.git/hooks/`:

1. **pre-commit** — Ruff lint, Ruff format, MyPy types, print()/debugger detection, secret scanning, file size
2. **commit-msg** — Validates conventional commit format
3. **pre-push** — Full linting, type checking, unit tests, coverage, security scan, import verification

Source templates live in `git-hooks/`. To reinstall: `./git-hooks/setup-hooks.sh`

Hooks auto-detect environment: `.venv/bin/` > Docker > global.

## Claude Code Hooks (Active)

Three Claude Code hooks in `.claude/hooks/`:

1. **validate-git-commit.sh** (PreToolUse) — Blocks `--no-verify`, `--force` push, and validates conventional commit format before Claude runs git commands
2. **post-git-ops.sh** (PostToolUse) — Logs commits to `implementation_plans/.commit-log-YYYY-MM-DD.txt` after successful git commits
3. **log-implementation.sh** (Stop) — Reminds to log implementation plans when >2 files changed in a session

## Implementation Plan Tracking

Every time a significant feature or change is implemented, **log it** in the `implementation_plans/` folder.

### Convention
- **Filename format:** `YYYY-MM-DD_HH-MM_<short-description>.md`
- **Example:** `2026-02-14_18-06_should-have-tier2-advanced-features.md`
- Include: objective, each phase with status, files created/modified, design decisions, test results
- Mark each phase as `[COMPLETED]`, `[IN PROGRESS]`, or `[PLANNED]`

### When to log
- After implementing a new feature or set of features
- After a major refactor
- After adding a new module or integration
- Any work that touches more than 2-3 files

## Development Environment

- Use `uv` for virtual environment management: `uv venv .venv && uv pip install -e ".[dev]"`
- Run tests: `.venv/bin/python -m pytest tests/unit/ -v`
- Never install packages globally — always use the `.venv`
- The `.venv/` is gitignored

## Architecture

- **ABC base classes** in `base.py` files for each module (document, retrieval, generation)
- **Pydantic configs** in `rag_sdk/config/config.py`
- **Strategy pattern** — orchestrator (`core.py`) selects implementations based on config
- **Composable patterns** — e.g., Corrective RAG wraps any retriever
- **All defaults preserve backward compatibility** — `recursive`/`dense`/`standard`
