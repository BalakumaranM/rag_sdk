#!/bin/bash
# Claude Code PostToolUse hook: track git operations and remind about implementation plans
# Runs after successful Bash calls involving git commit

INPUT=$(cat)
TOOL_NAME=$(echo "$INPUT" | jq -r '.tool_name // empty')
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty')

if [ "$TOOL_NAME" != "Bash" ]; then
    exit 0
fi

# After a successful git commit, log it to implementation tracking
if echo "$COMMAND" | grep -qE "git commit"; then
    PLAN_DIR="$CLAUDE_PROJECT_DIR/implementation_plans"
    TODAY=$(date "+%Y-%m-%d")
    COMMIT_MSG=$(cd "$CLAUDE_PROJECT_DIR" && git log -1 --pretty=%s 2>/dev/null)
    COMMIT_HASH=$(cd "$CLAUDE_PROJECT_DIR" && git log -1 --pretty=%h 2>/dev/null)

    if [ -n "$COMMIT_HASH" ]; then
        # Append to today's commit log
        COMMIT_LOG="$PLAN_DIR/.commit-log-${TODAY}.txt"
        echo "[$COMMIT_HASH] $(date '+%H:%M:%S') $COMMIT_MSG" >> "$COMMIT_LOG"

        cat <<EOF
{"additionalContext": "Git commit recorded: $COMMIT_HASH - $COMMIT_MSG. If this is part of a larger implementation, ensure it's documented in implementation_plans/."}
EOF
    fi
fi

exit 0
