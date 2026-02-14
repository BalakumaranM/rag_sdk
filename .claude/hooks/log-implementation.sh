#!/bin/bash
# Hook: log-implementation.sh
# Triggered on Stop event — reminds Claude to log implementation plans.
# Reads the stop event from stdin and outputs a system message reminder.

INPUT=$(cat)

# Check if implementation_plans directory exists
PLAN_DIR="$CLAUDE_PROJECT_DIR/implementation_plans"
if [ ! -d "$PLAN_DIR" ]; then
    mkdir -p "$PLAN_DIR"
fi

# Count files modified in this git session (uncommitted changes)
CHANGED_FILES=$(cd "$CLAUDE_PROJECT_DIR" && git diff --name-only HEAD 2>/dev/null | wc -l | tr -d ' ')
UNTRACKED_FILES=$(cd "$CLAUDE_PROJECT_DIR" && git ls-files --others --exclude-standard 2>/dev/null | wc -l | tr -d ' ')
TOTAL=$((CHANGED_FILES + UNTRACKED_FILES))

# Only remind if there were actual code changes (more than 2 files touched)
if [ "$TOTAL" -gt 2 ]; then
    # Get today's date for the plan filename
    TODAY=$(date "+%Y-%m-%d")
    EXISTING_PLANS=$(ls "$PLAN_DIR"/${TODAY}* 2>/dev/null | wc -l | tr -d ' ')

    cat <<EOF
{"additionalContext": "IMPLEMENTATION TRACKING REMINDER: $TOTAL files were changed/added in this session. If you implemented a feature or made significant changes, log them in the implementation_plans/ folder. Use the naming format: YYYY-MM-DD_HH-MM_<short-description>.md. There are currently $EXISTING_PLANS plan(s) logged today."}
EOF
fi

exit 0
