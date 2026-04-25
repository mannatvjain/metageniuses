# Metageniuses

Project for Apart Research's AI x Bio Hackathon.

## Stack
- **Language**:
- **Key dependencies**:

## Architecture

[ASCII diagram or short prose describing the system. Fill this in once the project shape is clear.]

## Conventions
- **No self-attribution**: Claude is a tool, not a person. Never add Co-Authored-By lines or credit Claude as a co-author in commits or anywhere else.
- **Owner tracking**: This is a multi-person project. Every commit must include the name of the human who instructed the Claude agent (e.g., `Instructed by: Mannat Jain` in the commit message). Ask if unclear.

## Dev commands
```bash
# [Fill in as commands are established.]
```

## Setup
This repo was cloned from `~/Developer/agent-scaffold`. After cloning, initialize it as a git repo:
```bash
cd ~/Developer/metageniuses
git init
git add -A
git commit -m "Initial scaffold from agent-scaffold template"
```

---

## Agent Protocol

You are one of many short-lived Claude sessions working on this project. The user relies on Claude to write code — knowledge must transfer between sessions via docs, not memory. You will not remember prior conversations. The docs are your memory.

### Before starting work
1. Read `INDEX.md` to understand the repo layout.
2. Read `PLAN.md` to see what's done and what's next.
3. Read `LOG.md` (latest entry) to understand where the last session left off.
4. Read any `*_REFERENCE.md` files listed in `INDEX.md` before writing code that touches those domains. Do not guess at APIs or platform behavior — it's documented there for a reason.
5. Read `CONTEXT.md` if it exists — it describes the problem domain and any external systems.

### During work

#### Planning (required for non-trivial tasks)
Before making changes, write a short ASCII plan and show it to the user:

```
+-------------------------------------+
| Task: <short description>           |
+-------------------------------------+
| 1. <step>                           |
| 2. <step>                           |
|    - <substep>                      |
| 3. <step>                           |
+-------------------------------------+
```

Wait for confirmation before proceeding. Keep plans concise.

#### Recap (required after completing each action)
After completing work, show an ASCII recap:

```
+-------------------------------------+
| Recap: <short description>          |
+-------------------------------------+
| Files edited:                       |
|  * path/to/file                     |
|    - <what changed>                 |
| Insights saved:                     |
|  > REFERENCE_DOC.md                 |
|    - <what was documented>          |
+-------------------------------------+
```

#### Documenting new knowledge
When you learn something non-obvious — a platform gotcha, an API quirk, a pattern that works — write it to the appropriate `*_REFERENCE.md` file immediately. If no reference doc exists for that domain yet, create one and add it to `INDEX.md`. This is how you pass knowledge to the next session.

### Handoff (end of every conversation)
When the user runs `/close` or the conversation is ending, complete this checklist:
1. Update any `*_REFERENCE.md` files with patterns learned this session.
2. Update `PLAN.md` — mark completed items `[x]`, add new items if the plan changed.
3. Append a dated entry to `LOG.md`: what changed, what's unfinished, what the next session should pick up.
4. Update `INDEX.md` if any files were added or removed.
5. If anything is half-finished, note it clearly in `LOG.md` so the next agent doesn't have to guess.

---

## File Guide

**This section describes what each file in the scaffold is for. After the initial setup session has populated the files below, delete this entire "File Guide" section** — it exists only to bootstrap the first session's understanding.

### INDEX.md
A flat map of every file in the repo with a one-line purpose. Two sections: **Documentation** and **Code** (add **Config** if needed). Use a markdown table:

```markdown
| File | Purpose |
|------|---------|
| `CLAUDE.md` | Project instructions, architecture, agent protocol |
| `INDEX.md` | This file — map of the repository |
```

Keep it updated. An agent should be able to read `INDEX.md` alone and know where everything is. No prose — just the table.

### PLAN.md
A checklist tracking every task in the project. Use `- [x]` for done, `- [ ]` for pending. Group by milestone or phase. Indent subtasks. Example:

```markdown
# Plan

## Phase 1: Foundation
- [x] Set up project structure
- [ ] Implement core data model
  - [x] Define schemas
  - [ ] Write migrations
- [ ] Build API layer
```

This is the single source of truth for "what needs to happen" and "what's already done." Every session should update it. Don't let it go stale.

### LOG.md
A reverse-chronological session journal. Each entry is dated and written by the agent at the end of a conversation. Format:

```markdown
### YYYY-MM-DD
- What was accomplished (bullet points, concise)
- What's unfinished and why
- What the next session should start with
- Any gotchas or decisions made that aren't obvious from the code
```

Newest entries at the top (after the header). This is how sessions hand off to each other. Be specific — "worked on the API" is useless; "added GET /messages endpoint, POST /messages still needs auth middleware" is useful.

### CONTEXT.md (created by first session)
Describes the problem domain and any external systems the project interacts with. This is everything an agent needs to understand *why* the code exists and *what* it connects to, without reading external repos or docs. Include:

- What problem this project solves
- External systems it talks to (APIs, databases, services) with enough detail to write code against them
- Data flows: what comes in, what goes out, what gets stored
- Domain vocabulary: terms that have specific meanings in this project
- Constraints: performance requirements, security boundaries, platform limitations

This file should be written once and updated rarely. If a section grows large enough to need its own file, split it into a `*_REFERENCE.md` and link from here.

### *_REFERENCE.md (created as needed)
Platform-specific or API-specific knowledge that agents discover through trial and error. Named after the domain: `FOUNDRY_REFERENCE.md`, `SIGNAL_CLI_REFERENCE.md`, `REACT_REFERENCE.md`, etc.

Each reference doc should contain:
- **Working patterns**: code snippets that actually work, with context
- **Gotchas**: things that don't work the way you'd expect, with the correct approach
- **API surface**: endpoints, parameters, return types — whatever the agent needs to avoid guessing

Format gotchas as:
```markdown
### Gotcha: [short description]
**Wrong**: [what you'd naively try]
**Right**: [what actually works]
**Why**: [one-line explanation]
```

These files grow over time. Every session that learns something new about the platform should append to the relevant reference doc. If you're about to search the web for how an API works, check the reference doc first — a previous session may have already figured it out.
