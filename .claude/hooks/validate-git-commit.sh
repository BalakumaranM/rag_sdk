#!/bin/bash
# Claude Code PreToolUse hook: validate git commit commands
# Ensures Claude follows conventional commit format and never skips git hooks

INPUT=$(cat)
TOOL_NAME=$(echo "$INPUT" | jq -r '.tool_name // empty')
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty')

# Only check Bash tool calls that involve git
if [ "$TOOL_NAME" != "Bash" ]; then
    exit 0
fi

# Block --no-verify on commits and pushes
if echo "$COMMAND" | grep -qE "git (commit|push).*--no-verify"; then
    echo "BLOCKED: Do not use --no-verify. Git hooks (pre-commit, commit-msg, pre-push) must always run. They enforce linting, type checking, conventional commits, and tests." >&2
    exit 2
fi

# Block --force push
if echo "$COMMAND" | grep -qE "git push.*--force|git push.*-f"; then
    echo "BLOCKED: Force push is not allowed. It can overwrite remote history. Use regular push or ask the user first." >&2
    exit 2
fi

# Validate conventional commit format on git commit -m
if echo "$COMMAND" | grep -qE "git commit"; then
    # Extract commit message from -m flag
    COMMIT_MSG=$(echo "$COMMAND" | grep -oP '(?<=-m\s")[^"]*|(?<=-m\s'"'"')[^'"'"']*' | head -1)

    # Also try HEREDOC pattern: cat <<'EOF'
    if [ -z "$COMMIT_MSG" ]; then
        COMMIT_MSG=$(echo "$COMMAND" | grep -oP "(?<=EOF\n).*(?=\n.*EOF)" | head -1)
    fi

    if [ -n "$COMMIT_MSG" ]; then
        FIRST_LINE=$(echo "$COMMIT_MSG" | head -1)

        # Skip merge commits
        if echo "$FIRST_LINE" | grep -qE "^Merge branch"; then
            exit 0
        fi

        # Check conventional commit format
        PATTERN="^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)(\([a-z0-9-]+\))?: .{1,100}$"
        if ! echo "$FIRST_LINE" | grep -qE "$PATTERN"; then
            cat >&2 <<ERRMSG
BLOCKED: Commit message does not follow Conventional Commits format.

Required format: <type>(<scope>): <description>

Valid types: feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert
Scope is optional but must be lowercase alphanumeric with hyphens.
Description must be 1-100 characters.

Examples:
  feat(retrieval): add Graph RAG retriever
  fix(config): handle missing API key gracefully
  test(generation): add CoVe unit tests
  refactor(core): extract init methods

Your message: "$FIRST_LINE"
ERRMSG
            exit 2
        fi
    fi
fi

exit 0
