# [Feature Name] Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** [One sentence describing what this builds]

**Architecture:** [2-3 sentences about approach]

**Tech Stack:** [Key technologies/libraries]

---

## Task 1: [Component Name]

**Files:**
- Create: `exact/path/to/file.ts`
- Modify: `exact/path/to/existing.ts:123-145`
- Test: `tests/exact/path/to/test.ts`

**Step 1: Write the failing test**

```typescript
describe('SpecificBehavior', () => {
  it('does expected thing', () => {
    const result = function(input);
    expect(result).toBe(expected);
  });
});
```

**Step 2: Run test to verify it fails**

Run: `npm test -- tests/path/test.ts`
Expected: FAIL with "function not defined"

**Step 3: Write minimal implementation**

```typescript
export function function(input: Type): ReturnType {
  return expected;
}
```

**Step 4: Run test to verify it passes**

Run: `npm test -- tests/path/test.ts`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/path/test.ts src/path/file.ts
git commit -m "feat: add specific feature"
```

---

## Task 2: [Next Component]

(Repeat structure...)

---

## Execution Options

**Plan complete and saved. Two execution options:**

1. **Subagent-Driven (this session)** - Fresh subagent per task, review between tasks
2. **Parallel Session (separate)** - Open new session with executing-plans skill

**Which approach?**
