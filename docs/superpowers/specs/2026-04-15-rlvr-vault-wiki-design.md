# RLVR Vault — LLM Wiki Design

**Date:** 2026-04-15  
**Scope:** Research-focused personal knowledge base for the RLVR-Comparison project, living in `RLVR_VAULT/`  
**Purpose:** Incrementally build and maintain a structured wiki over papers, experiment results, meeting notes, articles, and free-form notes — supporting both personal synthesis and writing outputs (lit reviews, comparison tables, paper sections)

---

## 1. Directory Structure

```
RLVR_VAULT/
  raw/
    papers/         # PDFs and markdown exports of papers
    results/        # experiment logs, eval outputs (symlinks or copies)
    clips/          # web-clipped articles (markdown from Obsidian Web Clipper)
    meetings/       # meeting notes, advisor conversations
    notes/          # free-form notes (e.g. migrated Notes.md)
    assets/         # images downloaded alongside clips

  wiki/
    concepts/       # one page per concept: ppo.md, grpo.md, advantage-estimation.md...
    entities/       # models, datasets, authors: llama-3-8b.md, gsm8k.md, humaneval.md...
    experiments/
      exp-2.7/      # exp-2.7-overview.md + exp-2.7-seed1.md, exp-2.7-seed2.md...
      exp-2.8/      # exp-2.8-overview.md + per-run pages (critic size variants)
      exp-2.9/      # exp-2.9-overview.md + per-run pages (label regime variants)
    sources/        # one summary page per raw source: schulman2017-ppo.md...
    outputs/        # generated artifacts: lit-review-draft.md, ppo-vs-grpo-table.md...

  index.md          # full catalog of all wiki pages, updated on every ingest
  log.md            # append-only timeline of ingests, queries, lint passes
  search.sh         # ripgrep wrapper for LLM to shell out to
  CLAUDE.md         # schema and rules governing this wiki agent
```

### File naming conventions
- All lowercase, hyphen-separated, no spaces
- Per-run experiment files: `exp-{id}-{seed|run|variant}.md`
- Source pages: `{firstauthor}{year}-{slug}.md` (e.g. `schulman2017-ppo.md`)
- Concept and entity pages: descriptive slug (e.g. `advantage-estimation.md`, `gsm8k.md`)

### Frontmatter (all wiki pages)
```yaml
---
type: concept | entity | experiment | source | output
tags: [ppo, advantage-estimation, ...]
sources: [schulman2017-ppo, shao2024-deepseek]   # backlinks to raw sources
updated: YYYY-MM-DD
---
```

---

## 2. Special Files

### `index.md`
Content-oriented catalog of every wiki page. Structure:
```markdown
# RLVR Vault Index

## Concepts
- [[ppo]] — Proximal Policy Optimization: clip-based policy gradient with a learned critic
- [[grpo]] — Group Relative Policy Optimization: critic-free, normalizes advantage within a group

## Entities
- [[gsm8k]] — Grade school math benchmark, 8.5k problems with verifiable answers
...

## Experiments
- [[exp-2.7-overview]] — Head-to-head: PPO vs GRPO vs DPO on GSM8K + HumanEval, 3 seeds
...

## Sources
- [[schulman2017-ppo]] — Proximal Policy Optimization Algorithms (Schulman et al. 2017)
...

## Outputs
- [[ppo-vs-grpo-table]] — Side-by-side comparison table: architecture, critic, advantage estimation
...
```

Updated on every ingest. The LLM reads this first on every query to identify relevant pages.

### `log.md`
Append-only chronological record. Each entry starts with a parseable prefix:
```markdown
## [YYYY-MM-DD] ingest | {source title}
## [YYYY-MM-DD] query | {question summary}
## [YYYY-MM-DD] lint | {summary of findings}
```

Enables: `grep "^\#\# \[" log.md | tail -10` to see recent activity.

---

## 3. `search.sh`

Thin ripgrep wrapper. Usage:
```bash
./search.sh "advantage estimation"           # search all wiki pages
./search.sh "advantage estimation" concepts  # scoped to a subdirectory
```

Returns matching file paths + lines. No external dependencies beyond `rg`.

---

## 4. `CLAUDE.md` Schema Contents

The schema file governs every wiki session. Sections:

1. **Identity & mandate** — "You are the RLVR wiki agent. Maintain `RLVR_VAULT/wiki/`. You read `raw/`; you never modify it."
2. **Directory map** — canonical reference for what lives where
3. **Naming & frontmatter spec** — as above
4. **Workflow: Ingest** — exact ordered steps (see Section 5)
5. **Workflow: Query** — exact ordered steps (see Section 5)
6. **Workflow: Lint** — exact ordered steps (see Section 5)
7. **Domain glossary** — seed definitions for: PPO, GRPO, DPO, RLVR, verifiable reward, advantage estimation, KL penalty, critic network, preference pairs, GSM8K, HumanEval
8. **Writing style** — terse, precise, citation-backed. No filler. Contradictions flagged with `> [!warning]` callouts. Uncertainty flagged with `> [!note] Uncertain:`.
9. **Output format rules**:
   - Comparison tables → markdown tables
   - Experiment result pages → fixed template: Config → Metrics → Key Findings → Links to raw
   - Lit review sections → prose with inline `[[source]]` citations
10. **Hard constraints** — never modify `raw/`; always update `index.md` after any write; always append to `log.md`; never leave orphan pages without an `index.md` entry

---

## 5. Workflows

### Ingest
Triggered when user drops a source in `raw/` and says "ingest X":

1. Read the source in full
2. Discuss key takeaways with the user (brief exchange)
3. Write `wiki/sources/{firstauthor}{year}-{slug}.md`
4. Update or create relevant `wiki/concepts/` pages — add new findings, flag contradictions
5. Update or create relevant `wiki/entities/` pages
6. If source is an experiment result: update the relevant per-run page in `wiki/experiments/exp-{id}/` and roll up to the overview page
7. Update `index.md` with any new pages
8. Append to `log.md`: `## [YYYY-MM-DD] ingest | {source title}`

Expected page touches: 8–12 for a paper; 3–5 for experiment results.

### Query
Triggered when user asks a question or requests an artifact:

1. Read `index.md` to identify relevant pages
2. Run `search.sh` for targeted lookup if needed
3. Read identified pages; synthesize answer
4. If the output is reusable (comparison table, lit review section, analysis), write it to `wiki/outputs/` and update `index.md`
5. Append to `log.md`: `## [YYYY-MM-DD] query | {question summary}`

### Lint
Triggered when user says "lint the wiki":

1. Scan for contradictions between pages
2. Identify orphan pages (no inbound links)
3. Identify concepts mentioned inline but lacking their own page
4. Flag stale claims superseded by newer sources
5. Suggest gaps: open questions, missing sources, underlinked pages
6. Append to `log.md`: `## [YYYY-MM-DD] lint | {summary of findings}`

---

## 6. Experiment Page Templates

### Per-run page (`exp-2.7-seed1.md`)
```markdown
---
type: experiment
tags: [exp-2.7, ppo, gsm8k, seed-1]
sources: []
updated: YYYY-MM-DD
---

# Exp 2.7 — Seed 1

## Config
- Model: LLaMA-3 8B
- Method: PPO
- Dataset: GSM8K
- Seed: 1
- Compute: ...

## Metrics
| Metric | Value |
|--------|-------|
| Accuracy | ... |
| Training stability | ... |
| Convergence step | ... |
| Advantage estimation error | ... |

## Key Findings
- ...

## Links
- Raw log: `../../../raw/results/exp-2.7-seed1.log`
```

### Experiment overview page (`exp-2.7-overview.md`)
```markdown
---
type: experiment
tags: [exp-2.7, ppo, grpo, dpo, gsm8k, humaneval]
sources: []
updated: YYYY-MM-DD
---

# Exp 2.7 — Head-to-Head Overview

## Summary
...

## Aggregate Results
| Method | GSM8K Acc (mean±std) | HumanEval Acc | Convergence |
|--------|----------------------|---------------|-------------|

## Per-run pages
- [[exp-2.7-seed1]], [[exp-2.7-seed2]], [[exp-2.7-seed3]]

## Key Findings
...

## Open Questions
...
```

---

## 7. Out of Scope (for now)
- Embedding-based search (qmd or similar) — index.md + search.sh sufficient at current scale; add later if needed
- Multi-user / team wiki — single-user for now
- Automated ingest (watch folder + trigger) — manual ingest workflow only
