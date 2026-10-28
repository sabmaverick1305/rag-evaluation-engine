---
name: ship-pr
description: Human-in-the-Loop (HITL) workflow to run compile/lint/test inspection, perform semantic review against specs, create a feature branch, and dispatch a GitHub PR.
---

# Ship PR - Human-In-The-Loop Developer Workflow

Use this skill whenever the user signals explicit approval to finalize code and open a Pull Request (e.g. *"Update the code and raise PR"*, *"Ship PR"*).

## Workflow Pipeline

```
[Human Signal: "Update code and ship PR"]
                    │
                    ▼
┌────────────────────────────────────────────────────────┐
│ Phase 1: Deterministic Inspector & Quality Gate        │
│ - Runs: scripts/verify_and_review.py                   │
│   1. python3 -m py_compile src/                        │
│   2. ruff check src/ tests/                            │
│   3. pytest tests/                                     │
└───────────────────┬────────────────────────────────────┘
                    │
                    ├─► [Failure] ──► Auto-repair loop (up to 3x)
                    │
                    ▼ [All Gates Passed]
┌────────────────────────────────────────────────────────┐
│ Phase 2: Semantic Code Review                          │
│ - Inspects git diff against main                       │
│ - Validates compliance with CLAUDE.md                  │
│ - Audits requirements against specs/<feature>/spec.md  │
└───────────────────┬────────────────────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────────────────────┐
│ Phase 3: Branch Isolation & Git Commit                 │
│ - Ensures active branch is feat/<feature-name>         │
│ - Creates conventional commit                          │
└───────────────────┬────────────────────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────────────────────┐
│ Phase 4: GitHub PR Dispatcher                          │
│ - Pushes branch to origin                              │
│ - Runs: gh pr create --title "..." --body "..."        │
│ - Returns PR URL for final human merge sign-off        │
└────────────────────────────────────────────────────────┘
```

---

## Step-by-Step Instructions

### Step 1: Execute Pre-PR Inspection Gate
Run the automated verification script:
```bash
python3 scripts/verify_and_review.py
```

* **If failures occur**:
  1. Inspect the error output (syntax error, lint violation, or failing unit test).
  2. Implement the targeted fix in `src/` or `tests/`.
  3. Re-run `python3 scripts/verify_and_review.py` until exit code is 0.
  4. Do not proceed to Step 2 until all quality gates pass cleanly.

### Step 2: Semantic Code Review
1. Review the git diff against `main`:
   ```bash
   git diff --stat origin/main
   ```
2. Verify:
   - Does this change adhere to [`CLAUDE.md`](file:///Users/sabyasachi/rag-evaluation-engine/CLAUDE.md) (strict Pydantic models, typed signatures, hermetic tests)?
   - Are acceptance criteria in the corresponding `specs/<feature>/spec.md` satisfied?
   - Are there any leftover debugging prints, dead code, or unmocked network calls?

### Step 3: Git Branch & Commit Hygiene
1. Determine feature name from active spec (e.g., `langgraph-agent-workflow`).
2. Create and switch to a dedicated feature branch if currently on `main`:
   ```bash
   git checkout -b feat/<feature-name>
   ```
3. Stage changes and commit:
   ```bash
   git add src/ tests/ pyproject.toml specs/ skills/ scripts/
   git commit -m "feat: <clear summary of feature conforming to spec>"
   ```

### Step 4: Dispatch GitHub Pull Request
1. Push branch to remote:
   ```bash
   git push -u origin feat/<feature-name>
   ```
2. Generate PR using GitHub CLI:
   ```bash
   gh pr create \
     --title "feat: <Feature Title>" \
     --body "$(cat <<EOF
   ## Summary
   Brief overview of what was implemented.

   ## Spec Reference
   Closes / Implements: \`specs/<feature>/spec.md\`

   ## Verification & Quality Gates
   - [x] Syntax & Compilation (\`py_compile\`) passed
   - [x] Ruff Lint & Formatting passed
   - [x] Unit Test Suite passed hermetically (0 network calls)

   ## Architecture Notes
   Key design decisions or trade-offs made during implementation.
   EOF
   )"
   ```
3. Output the created PR URL to the user for final review and merge.

