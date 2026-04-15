# RLVR Vault Wiki Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Set up `RLVR_VAULT/` as a fully operational LLM wiki with directory structure, schema, working search, and seed knowledge pages covering all core concepts, entities, and experiment skeletons.

**Architecture:** Concept-first layered directory structure inside the existing Obsidian vault at `RLVR_VAULT/`. `CLAUDE.md` governs wiki agent behavior. `index.md` + `search.sh` provide navigation. All wiki pages are markdown with YAML frontmatter. `log.md` is append-only.

**Tech Stack:** Markdown, YAML frontmatter, ripgrep (`rg`), bash

---

### Task 1: Create directory structure

**Files:**
- Create: `RLVR_VAULT/raw/{papers,results,clips,meetings,notes,assets}/`
- Create: `RLVR_VAULT/wiki/{concepts,entities,sources,outputs}/`
- Create: `RLVR_VAULT/wiki/experiments/{exp-2.7,exp-2.8,exp-2.9}/`

- [ ] **Step 1: Verify rg is available**

```bash
rg --version
```

Expected: `ripgrep X.X.X` (any version)

- [ ] **Step 2: Create all directories**

```bash
mkdir -p RLVR_VAULT/raw/{papers,results,clips,meetings,notes,assets}
mkdir -p RLVR_VAULT/wiki/{concepts,entities,sources,outputs}
mkdir -p RLVR_VAULT/wiki/experiments/{exp-2.7,exp-2.8,exp-2.9}
```

- [ ] **Step 3: Verify structure**

```bash
find RLVR_VAULT -type d | grep -v '.obsidian' | sort
```

Expected (all present):
```
RLVR_VAULT/raw/assets
RLVR_VAULT/raw/clips
RLVR_VAULT/raw/meetings
RLVR_VAULT/raw/notes
RLVR_VAULT/raw/papers
RLVR_VAULT/raw/results
RLVR_VAULT/wiki/concepts
RLVR_VAULT/wiki/entities
RLVR_VAULT/wiki/experiments/exp-2.7
RLVR_VAULT/wiki/experiments/exp-2.8
RLVR_VAULT/wiki/experiments/exp-2.9
RLVR_VAULT/wiki/outputs
RLVR_VAULT/wiki/sources
```

- [ ] **Step 4: Commit**

```bash
git add RLVR_VAULT/
git commit -m "feat: scaffold RLVR_VAULT directory structure"
```

---

### Task 2: Create search.sh

**Files:**
- Create: `RLVR_VAULT/search.sh`

- [ ] **Step 1: Create search.sh**

```bash
#!/usr/bin/env bash
# Usage:
#   ./search.sh "query"           -- search all wiki pages
#   ./search.sh "query" concepts  -- scoped to wiki/concepts/
#
# Returns file paths and matching lines. Exits 0 on match, 1 on no match (rg default).

set -euo pipefail

QUERY="${1:?Usage: search.sh <query> [subdir]}"
SUBDIR="${2:-}"
WIKI_DIR="$(dirname "$0")/wiki"

if [[ -n "$SUBDIR" ]]; then
  TARGET="$WIKI_DIR/$SUBDIR"
else
  TARGET="$WIKI_DIR"
fi

if [[ ! -d "$TARGET" ]]; then
  echo "Error: directory not found: $TARGET" >&2
  exit 1
fi

rg --color=never --heading --line-number "$QUERY" "$TARGET"
```

- [ ] **Step 2: Make executable**

```bash
chmod +x RLVR_VAULT/search.sh
```

- [ ] **Step 3: Verify script is valid (no wiki pages yet, rg exits 1 on no match — suppress that)**

```bash
cd RLVR_VAULT && bash search.sh "ppo" 2>&1 || true
cd ..
```

Expected: no output (no wiki files yet). No bash syntax error.

- [ ] **Step 4: Commit**

```bash
git add RLVR_VAULT/search.sh
git commit -m "feat: add search.sh ripgrep wrapper"
```

---

### Task 3: Create log.md

**Files:**
- Create: `RLVR_VAULT/log.md`

- [ ] **Step 1: Create log.md**

```markdown
# RLVR Vault — Activity Log

Append-only. Each entry: `## [YYYY-MM-DD] {type} | {title}`
Types: `ingest`, `query`, `lint`

Parse recent entries: `grep "^## \[" log.md | tail -10`

---

## [2026-04-15] ingest | Wiki initialization — seed pages and schema
```

- [ ] **Step 2: Verify**

```bash
grep "^## \[" RLVR_VAULT/log.md
```

Expected: `## [2026-04-15] ingest | Wiki initialization — seed pages and schema`

- [ ] **Step 3: Commit**

```bash
git add RLVR_VAULT/log.md
git commit -m "feat: initialize log.md"
```

---

### Task 4: Create CLAUDE.md schema

**Files:**
- Create: `RLVR_VAULT/CLAUDE.md`

This is the most important file. It governs all future wiki agent behavior. Write it verbatim as shown.

- [ ] **Step 1: Create CLAUDE.md**

The file content uses markdown with nested code blocks. Write each section carefully. The complete content:

~~~markdown
# RLVR Vault — Wiki Agent Schema

## Identity & Mandate

You are the RLVR wiki agent for this research project (comparing PPO, GRPO, DPO on RLVR tasks).
Your job is to build and maintain the structured wiki in `RLVR_VAULT/wiki/`.

**Hard constraints — never violate these:**
1. You READ from `RLVR_VAULT/raw/` — you NEVER modify or delete anything in `raw/`
2. You ALWAYS update `index.md` after creating or significantly updating any wiki page
3. You ALWAYS append to `log.md` after completing any operation (ingest, query, lint)
4. You NEVER leave a wiki page without an `index.md` entry
5. Every wiki page MUST have valid YAML frontmatter (see spec below)
6. Contradictions between pages MUST be flagged with `> [!warning]` callouts — never silently overwrite

---

## Directory Map

```
RLVR_VAULT/
  raw/                    # IMMUTABLE source collection
    papers/               # PDFs and markdown exports of papers
    results/              # experiment logs, eval outputs
    clips/                # web-clipped articles (Obsidian Web Clipper → markdown)
    meetings/             # meeting notes, advisor conversations
    notes/                # free-form notes
    assets/               # images downloaded alongside clips

  wiki/                   # LLM-maintained wiki (you write here)
    concepts/             # one page per concept or method
    entities/             # models, datasets, authors, tools
    experiments/
      exp-2.7/            # head-to-head experiment pages
      exp-2.8/            # critic sweep pages
      exp-2.9/            # label regime pages
    sources/              # one summary page per raw source ingested
    outputs/              # generated artifacts: tables, drafts, analyses

  index.md                # content-oriented catalog of all wiki pages
  log.md                  # append-only activity timeline
  search.sh               # ripgrep wrapper: ./search.sh "query" [subdir]
  CLAUDE.md               # this file
```

---

## File Naming Conventions

- All lowercase, hyphen-separated, no spaces or underscores
- **Concept/entity pages:** descriptive slug — `advantage-estimation.md`, `gsm8k.md`
- **Source pages:** `{firstauthor}{year}-{slug}.md` — `schulman2017-ppo.md`
- **Per-run experiment pages:** `exp-{id}-{seed|run|variant}.md` — `exp-2.7-seed1.md`, `exp-2.8-critic-large.md`
- **Experiment overviews:** `exp-{id}-overview.md`
- **Output pages:** descriptive slug — `ppo-vs-grpo-comparison.md`

---

## Frontmatter Specification

Every wiki page must begin with this YAML block:

```yaml
---
type: concept | entity | experiment | source | output
tags: [tag1, tag2, ...]
sources: [firstauthor-year-slug, ...]   # backlinks to wiki/sources/ pages
updated: YYYY-MM-DD
---
```

`type` must be exactly one of: `concept`, `entity`, `experiment`, `source`, `output`.
`sources` is an empty list `[]` until populated by ingest.

---

## Workflow: Ingest

**Trigger:** User says "ingest X" where X is a file in `raw/`.

**Steps (in order, do not skip):**

1. Read the full source file. If PDF, read the text. Note any images for separate viewing.
2. Briefly discuss 3–5 key takeaways with the user. Ask if they want to emphasize anything.
3. Write `wiki/sources/{firstauthor}{year}-{slug}.md` using the Source Page Template below.
4. For each concept mentioned: update or create `wiki/concepts/` page. Add new findings. Flag contradictions with `> [!warning] Contradiction with [[other-page]]: {describe conflict}`.
5. For each entity mentioned (model, dataset, author): update or create `wiki/entities/` page.
6. If source is an experiment result: update or create the per-run page in `wiki/experiments/exp-{id}/` using the Experiment Run Template. Then update the experiment overview page.
7. Update `index.md`: add entries for new pages; update one-line summaries for significantly changed pages.
8. Append to `log.md`: `## [YYYY-MM-DD] ingest | {source title}`

**Expected page touches:** 8–12 for a paper; 3–5 for experiment results; 2–4 for notes/clips.

---

## Workflow: Query

**Trigger:** User asks a question or requests an artifact.

**Steps (in order):**

1. Read `index.md` to identify relevant pages.
2. Run `./search.sh "key term" [subdir]` if the index scan is inconclusive.
3. Read identified pages. Synthesize answer grounded in wiki content.
4. If the output is reusable (comparison table, lit review paragraph, analysis):
   - Write it to `wiki/outputs/{slug}.md` using the Output Page Template.
   - Add it to `index.md` under **Outputs**.
5. Append to `log.md`: `## [YYYY-MM-DD] query | {question summary}`

---

## Workflow: Lint

**Trigger:** User says "lint the wiki".

**Steps (in order):**

1. Read `index.md` in full.
2. For each page listed: read it and check for contradictions with other pages you've read. Flag any with `> [!warning]`.
3. Identify orphan pages: pages in `wiki/` with no inbound `[[links]]` from other pages.
4. Identify concepts mentioned inline that lack their own page in `wiki/concepts/`.
5. Flag stale claims: compare claims across source dates; note when newer sources supersede older ones.
6. Output a lint report listing: contradictions, orphans, missing pages, stale claims, suggested new sources.
7. Append to `log.md`: `## [YYYY-MM-DD] lint | {N contradictions, M orphans, K gaps found}`

---

## Page Templates

### Source Page Template

```markdown
---
type: source
tags: [tag1, tag2]
sources: []
updated: YYYY-MM-DD
---

# {Title} ({Author} et al., {Year})

**Citation:** {Author} et al. ({Year}). {Title}. {Venue}.
**Raw file:** `../../../raw/{subdir}/{filename}`

## Summary
{2–4 sentence summary of the core contribution}

## Key Claims
- {claim 1} — supported by {evidence}
- {claim 2}

## Relevance to This Project
{How this source informs PPO/GRPO/DPO comparison or specific experiments}

## Connections
- [[concept-page]] — {how this source relates}
- [[entity-page]]
```

### Concept Page Template

```markdown
---
type: concept
tags: [tag1, tag2]
sources: []
updated: YYYY-MM-DD
---

# {Concept Name}

{2–3 sentence definition. Precise and citation-ready.}

## Mechanism
{How it works — equations, pseudocode, or prose as appropriate}

## In This Project
{How this concept manifests in PPO/GRPO/DPO implementations here}

## Connections
- [[related-concept]]

## Key Sources
- [[firstauthor-year-slug]] — {one-line relevance}
```

### Experiment Run Template

```markdown
---
type: experiment
tags: [exp-{id}, {method}, {dataset}, {variant}]
sources: []
updated: YYYY-MM-DD
---

# Exp {id} — {Variant Description}

## Config
| Parameter | Value |
|-----------|-------|
| Model | LLaMA-3 8B |
| Method | {PPO/GRPO/DPO} |
| Dataset | {GSM8K/HumanEval} |
| Seed | {N} |
| Compute | {GPU-hours or steps} |
| Key hyperparams | {lr, batch size, etc.} |

## Metrics
| Metric | Value |
|--------|-------|
| Accuracy | |
| Training stability | |
| Convergence step | |
| Advantage estimation error | |

## Key Findings
- {finding 1}

## Links
- Raw log: `../../../raw/results/{filename}`
```

### Experiment Overview Template

```markdown
---
type: experiment
tags: [exp-{id}, {all methods}, {all datasets}]
sources: []
updated: YYYY-MM-DD
---

# Exp {id} — {Title}

## Goal
{One sentence.}

## Setup
{Methods compared, datasets, conditions, compute budget}

## Aggregate Results
| {dimension} | {metric 1} | {metric 2} |
|-------------|-----------|-----------|

## Per-Run Pages
- [[exp-{id}-{variant1}]], [[exp-{id}-{variant2}]]

## Key Findings
- {finding}

## Open Questions
- {question}
```

### Output Page Template

```markdown
---
type: output
tags: [tag1, tag2]
sources: [firstauthor-year-slug, ...]
updated: YYYY-MM-DD
---

# {Output Title}

**Generated in response to:** {query or request}

{content — table, prose, analysis, etc.}
```

---

## Domain Glossary

Seed definitions. Extend or correct as sources are ingested.

**PPO (Proximal Policy Optimization):** Policy gradient method using a clipped surrogate objective to limit policy update size. Requires a learned critic (value network) for advantage estimation via GAE. Standard in RLHF pipelines. [[ppo]]

**GRPO (Group Relative Policy Optimization):** Critic-free variant from DeepSeek-R1. Computes advantages by normalizing rewards within a group of responses to the same prompt: `A_i = (r_i - mean(r)) / std(r)`. No separate value network. [[grpo]]

**DPO (Direct Preference Optimization):** Bypasses RL. Optimizes policy directly from (prompt, chosen, rejected) preference pairs via a supervised objective from the Bradley-Terry model. In RLVR: preference pairs generated synthetically from verifiable rewards. [[dpo]]

**RLVR (Reinforcement Learning from Verifiable Rewards):** RL paradigm where rewards come from checking output correctness (math answers, code execution) rather than a learned reward model. Eliminates reward model training and reward hacking. [[rlvr]]

**Advantage estimation:** `A(s,a) = Q(s,a) - V(s)`. In PPO: via GAE using the critic. In GRPO: via group reward normalization without a critic. [[advantage-estimation]]

**KL penalty:** `KL(π_θ || π_ref)` regularization preventing policy drift from the base model. Controlled by coefficient β. Present in all three methods. [[kl-penalty]]

**Verifiable reward:** Binary/scalar reward from checking output correctness without a learned model. GSM8K: numeric answer match. HumanEval: code execution against unit tests. [[verifiable-reward]]

**Preference pairs:** (prompt, chosen, rejected) triples for DPO. In this project: synthetically generated by sampling multiple rollouts per prompt and ranking by verifiable reward. [[preference-pairs]]

**Critic network:** Value network V(s) for PPO advantage estimation. Swept across sizes (none/small/medium/large) in exp 2.8. MC advantage error = `|V(s) - MC_return|` averaged over rollouts. [[critic-network]]

**GSM8K:** 8,500 grade-school math problems with verifiable numeric answers. Train/test: 7,473/1,319. [[gsm8k]]

**HumanEval:** 164 Python programming problems verified by unit test execution. pass@k metric. [[humaneval]]

---

## Writing Style

- **Terse and precise.** Every sentence earns its place. No filler.
- **Citation-backed.** Claims on concept/entity pages link to `[[source-page]]`.
- **Contradictions flagged:** `> [!warning] Contradiction with [[page]]: {describe the conflict}`
- **Uncertainty flagged:** `> [!note] Uncertain: {what is uncertain and why}`
- **No first-person** in wiki pages. Write as a reference document.
- **Prose in outputs.** Tables for comparisons. Code blocks for pseudocode/equations.

---

## Output Format Rules

| Query type | Format |
|------------|--------|
| Method comparison | Markdown table (rows = methods, cols = dimensions) |
| Experiment results | Experiment Run or Overview template |
| Literature review section | Prose with inline `[[source]]` citations, ~200–400 words |
| Concept explanation | Concept Page template or inline prose |
| Analysis / synthesis | Prose filed to `wiki/outputs/` |

---

## Hard Constraints (repeated for emphasis)

- Never modify anything in `raw/`
- Always update `index.md` after any wiki write
- Always append to `log.md` after any operation
- Never create an orphan page (no `index.md` entry)
- Every wiki page must have valid YAML frontmatter
~~~

- [ ] **Step 2: Verify CLAUDE.md was written correctly**

```bash
wc -l RLVR_VAULT/CLAUDE.md
```

Expected: 200+ lines

```bash
grep "## Workflow: Ingest" RLVR_VAULT/CLAUDE.md
grep "## Domain Glossary" RLVR_VAULT/CLAUDE.md
grep "Hard constraints" RLVR_VAULT/CLAUDE.md
```

Expected: each grep returns exactly one match.

- [ ] **Step 3: Commit**

```bash
git add RLVR_VAULT/CLAUDE.md
git commit -m "feat: add CLAUDE.md wiki agent schema"
```

---

### Task 5: Create seed concept pages (9 pages)

**Files:**
- Create: `RLVR_VAULT/wiki/concepts/ppo.md`
- Create: `RLVR_VAULT/wiki/concepts/grpo.md`
- Create: `RLVR_VAULT/wiki/concepts/dpo.md`
- Create: `RLVR_VAULT/wiki/concepts/rlvr.md`
- Create: `RLVR_VAULT/wiki/concepts/advantage-estimation.md`
- Create: `RLVR_VAULT/wiki/concepts/kl-penalty.md`
- Create: `RLVR_VAULT/wiki/concepts/verifiable-reward.md`
- Create: `RLVR_VAULT/wiki/concepts/preference-pairs.md`
- Create: `RLVR_VAULT/wiki/concepts/critic-network.md`

- [ ] **Step 1: Create ppo.md**

```markdown
---
type: concept
tags: [ppo, policy-gradient, rl, critic, advantage-estimation]
sources: []
updated: 2026-04-15
---

# PPO (Proximal Policy Optimization)

Policy gradient method using a clipped surrogate objective to prevent excessively large policy updates. Requires a learned critic (value network) to estimate advantages via GAE.

## Mechanism

**Clipped objective:**

    L_CLIP = E[min(r_t(θ) · A_t, clip(r_t(θ), 1-ε, 1+ε) · A_t)]

where `r_t(θ) = π_θ(a|s) / π_θ_old(a|s)` is the probability ratio and ε is the clip range (typically 0.2).

**Advantage estimation (GAE):**

    A_t = Σ (γλ)^k · δ_{t+k}   where δ_t = r_t + γV(s_{t+1}) - V(s_t)

**Value loss:** `L_VF = E[(V(s_t) - R_t)^2]`

## In This Project

Used in exp 2.7 (head-to-head) and exp 2.8 (critic size sweep). Built on LLaMA-3 8B with a modular critic. Advantage estimation error tracked to compare against GRPO's critic-free approach.

## Connections
- [[grpo]] — critic-free alternative; compared in exp 2.7 and 2.8
- [[advantage-estimation]] — core mechanism
- [[critic-network]] — required component
- [[kl-penalty]] — added as regularization
- [[rlvr]] — reward signal source

## Key Sources
_(populated on paper ingest)_
```

- [ ] **Step 2: Create grpo.md**

```markdown
---
type: concept
tags: [grpo, policy-gradient, rl, critic-free, advantage-estimation]
sources: []
updated: 2026-04-15
---

# GRPO (Group Relative Policy Optimization)

Critic-free policy optimization from DeepSeek-R1. Computes advantages by normalizing rewards within a group of responses to the same prompt, eliminating the need for a separate value network.

## Mechanism

For G responses `{o_1, ..., o_G}` sampled from the same prompt:

    A_i = (r_i - mean(r)) / std(r)

**Objective:**

    L_GRPO = E[min(r_t(θ) · A_i, clip(r_t(θ), 1-ε, 1+ε) · A_i)] - β · KL(π || π_ref)

No value network trained. Lower memory footprint than PPO (no critic parameters or optimizer states).

## In This Project

Baseline "no critic" condition in exp 2.8. Head-to-head with PPO and DPO in exp 2.7. Compared against DPO under sparse and noisy labels in exp 2.9.

## Connections
- [[ppo]] — compared in exp 2.7 and 2.8
- [[advantage-estimation]] — group-normalized variant
- [[kl-penalty]] — explicit in GRPO objective
- [[verifiable-reward]] — reward signal for group ranking
- [[dpo]] — compared in exp 2.9

## Key Sources
_(populated on paper ingest)_
```

- [ ] **Step 3: Create dpo.md**

```markdown
---
type: concept
tags: [dpo, preference-optimization, supervised, preference-pairs]
sources: []
updated: 2026-04-15
---

# DPO (Direct Preference Optimization)

Supervised method that directly optimizes a language model from preference pairs without explicit RL. Derived from the Bradley-Terry preference model and the optimal policy under a KL-constrained reward objective.

## Mechanism

**Objective:**

    L_DPO = -E[log σ(β · (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))]

where `y_w` = chosen response, `y_l` = rejected response, β controls KL strength.

No reward model, no online sampling, no critic. Purely supervised on a preference dataset.

## In This Project

Built on LLaMA-3 8B. Preference pairs generated synthetically from verifiable rewards (correct = chosen, incorrect = rejected). Compared against GRPO under full, sparse (10%), and noisy (10% flipped) label conditions in exp 2.9.

> [!note] Uncertain: Whether synthetic pair generation quality significantly limits DPO performance relative to human-labeled preferences. To be investigated in exp 2.9.

## Connections
- [[preference-pairs]] — required input format
- [[grpo]] — compared in exp 2.9
- [[verifiable-reward]] — constructs preference pairs
- [[kl-penalty]] — implicit in β term

## Key Sources
_(populated on paper ingest)_
```

- [ ] **Step 4: Create rlvr.md**

```markdown
---
type: concept
tags: [rlvr, reward-signal, verifiable-reward, rl]
sources: []
updated: 2026-04-15
---

# RLVR (Reinforcement Learning from Verifiable Rewards)

RL training paradigm where reward signals are computed by checking output correctness against ground truth — math answers, code execution results — rather than a learned reward model.

## Mechanism

1. Model generates a response to a prompt
2. Response is checked against a verifiable criterion (numeric answer match, code passes unit tests)
3. Binary or scalar reward returned: 1.0 (correct) or 0.0 (incorrect)
4. Reward used directly as training signal

No reward model training required. Reward hacking via reward model overoptimization is eliminated by design.

## In This Project

Unifying reward framework for all three methods. PPO and GRPO use verifiable rewards online (per rollout). DPO uses them offline to construct preference pairs.

## Connections
- [[verifiable-reward]] — the reward computation mechanism
- [[ppo]], [[grpo]], [[dpo]] — all use RLVR as reward signal
- [[gsm8k]], [[humaneval]] — the two verifiable benchmarks

## Key Sources
_(populated on paper ingest)_
```

- [ ] **Step 5: Create advantage-estimation.md**

```markdown
---
type: concept
tags: [advantage-estimation, gae, critic, ppo, grpo]
sources: []
updated: 2026-04-15
---

# Advantage Estimation

`A(s,a) = Q(s,a) - V(s)` measures how much better action `a` is than the expected action at state `s`. Central to all policy gradient methods.

## Mechanism

**PPO (GAE):**

    δ_t = r_t + γV(s_{t+1}) - V(s_t)
    A_t^GAE = Σ_{k=0}^{T} (γλ)^k · δ_{t+k}

Requires trained critic V(s). λ controls bias-variance tradeoff.

**GRPO (group normalization):**

    A_i = (r_i - mean(r_group)) / std(r_group)

No critic. Higher variance per estimate; unbiased given sufficient group size.

**Monte Carlo advantage error (exp 2.8 metric):**

    error = mean(|V(s) - MC_return|) over rollouts

Tracks how accurately the critic approximates the true return. Swept across critic sizes in exp 2.8.

## In This Project

Key comparison axis in exp 2.8: does a larger critic reduce advantage estimation error enough to justify compute cost vs GRPO's zero-critic baseline?

## Connections
- [[ppo]] — GAE-based estimation
- [[grpo]] — group-normalization-based estimation
- [[critic-network]] — required for GAE
- [[rlvr]] — reward signal feeding estimates

## Key Sources
_(populated on paper ingest)_
```

- [ ] **Step 6: Create kl-penalty.md**

```markdown
---
type: concept
tags: [kl-penalty, regularization, policy-constraint, ppo, grpo, dpo]
sources: []
updated: 2026-04-15
---

# KL Penalty

KL divergence `KL(π_θ || π_ref)` between the current policy and a reference policy (typically the SFT/base model). Regularization to prevent policy drift from the pretrained distribution.

## Mechanism

**Soft KL penalty (added to reward):**

    r'(s,a) = r(s,a) - β · log(π_θ(a|s) / π_ref(a|s))

β is the KL coefficient — higher β keeps the policy closer to π_ref.

PPO's ratio clipping is a related proxy that limits per-step policy change without an explicit KL term.

## In This Project

Applied explicitly in GRPO objective. PPO uses clipping as a proxy. DPO has an implicit KL term controlled by β. Kept matched across all three methods in exp 2.7 for fair comparison.

## Connections
- [[ppo]], [[grpo]], [[dpo]] — all incorporate KL constraint
- [[rlvr]] — KL penalty balances reward maximization against reference drift

## Key Sources
_(populated on paper ingest)_
```

- [ ] **Step 7: Create verifiable-reward.md**

```markdown
---
type: concept
tags: [verifiable-reward, reward-signal, rlvr, gsm8k, humaneval]
sources: []
updated: 2026-04-15
---

# Verifiable Reward

Scalar reward computed by checking model output correctness against ground truth, without a learned reward model.

## Mechanism

**GSM8K:** Extract final numeric answer from model response (regex/parser). Compare to ground truth. Reward = 1.0 if match, 0.0 otherwise.

**HumanEval:** Execute generated Python code against provided unit tests. Reward = fraction of tests passed (or binary pass@1).

No reward model parameters. No reward model training. Reward cannot be hacked via distributional shift of a reward model.

## In This Project

Primary reward signal for PPO and GRPO (online). Used offline to rank rollouts and construct preference pairs for DPO.

## Connections
- [[rlvr]] — the broader paradigm
- [[gsm8k]], [[humaneval]] — verifiable benchmark sources
- [[preference-pairs]] — downstream use for DPO
- [[grpo]] — uses group of verifiable rewards to compute advantages

## Key Sources
_(populated on paper ingest)_
```

- [ ] **Step 8: Create preference-pairs.md**

```markdown
---
type: concept
tags: [preference-pairs, dpo, synthetic-data, rlvr]
sources: []
updated: 2026-04-15
---

# Preference Pairs

(prompt, chosen, rejected) triples required by DPO. In this project, generated synthetically from verifiable rewards rather than human annotation.

## Mechanism

**Synthetic generation pipeline:**
1. For each prompt, sample G rollouts from the current (or SFT) policy
2. Score each rollout with the verifiable reward function
3. Pair highest-reward response (chosen) with lowest-reward response (rejected)
4. Filter pairs where chosen == rejected reward (ambiguous signal)

G (group size) is a hyperparameter affecting pair quality and diversity.

## In This Project

Built as a standalone pipeline. Used to construct the DPO training dataset for exp 2.9. Quality of synthetic pairs is a potential confounder when comparing DPO to GRPO under sparse label conditions.

> [!note] Uncertain: Whether pairs generated under sparse labels (10% of prompts have any correct rollout) provide sufficient coverage for stable DPO training.

## Connections
- [[dpo]] — consumes preference pairs
- [[verifiable-reward]] — scoring mechanism
- [[grpo]] — alternative that doesn't require pairs
- [[rlvr]] — reward source

## Key Sources
_(populated on paper ingest)_
```

- [ ] **Step 9: Create critic-network.md**

```markdown
---
type: concept
tags: [critic-network, value-network, ppo, advantage-estimation, exp-2.8]
sources: []
updated: 2026-04-15
---

# Critic Network

Value network V(s) estimating expected cumulative reward from a given state. Required by PPO for advantage estimation via GAE. Not used by GRPO or DPO.

## Mechanism

**Architecture:** Separate model head on top of the policy backbone, or a smaller separate transformer.

**Training objective:** `L_VF = E[(V(s_t) - R_t)^2]` where R_t is the discounted Monte Carlo return.

**Sizes swept in exp 2.8:**
- None — GRPO baseline (no critic)
- Small — shallow MLP head on frozen policy (~10M params)
- Medium — full LM head fine-tuned (~800M params)
- Large — separate transformer (~8B params)

**Monte Carlo advantage error:**

    MC_error = mean(|V(s_t) - MC_return_t|) over rollouts

Key metric in exp 2.8 crossover plot: critic error vs accuracy.

## In This Project

Central variable in exp 2.8. Hypothesis: larger critics reduce advantage estimation error, improving sample efficiency — but compute cost may not justify the gain vs GRPO's zero-cost baseline.

## Connections
- [[ppo]] — requires critic for GAE
- [[advantage-estimation]] — uses critic output
- [[grpo]] — critic-free baseline for comparison

## Key Sources
_(populated on paper ingest)_
```

- [ ] **Step 10: Verify all 9 concept pages exist with frontmatter**

```bash
ls RLVR_VAULT/wiki/concepts/
```

Expected: `advantage-estimation.md  critic-network.md  dpo.md  grpo.md  kl-penalty.md  ppo.md  preference-pairs.md  rlvr.md  verifiable-reward.md`

```bash
for f in RLVR_VAULT/wiki/concepts/*.md; do
  head -1 "$f" | grep -q "^---$" || echo "MISSING FRONTMATTER: $f"
done
```

Expected: no output (all files have frontmatter).

- [ ] **Step 11: Commit**

```bash
git add RLVR_VAULT/wiki/concepts/
git commit -m "feat: add 9 seed concept pages (PPO, GRPO, DPO, RLVR, core concepts)"
```

---

### Task 6: Create seed entity pages (3 pages)

**Files:**
- Create: `RLVR_VAULT/wiki/entities/llama-3-8b.md`
- Create: `RLVR_VAULT/wiki/entities/gsm8k.md`
- Create: `RLVR_VAULT/wiki/entities/humaneval.md`

- [ ] **Step 1: Create llama-3-8b.md**

```markdown
---
type: entity
tags: [llama, meta, base-model, transformer]
sources: []
updated: 2026-04-15
---

# LLaMA-3 8B

Meta's LLaMA-3 8B parameter open-weight language model. Base policy model for all three alignment methods in this project.

## Key Properties
- Parameters: 8 billion
- Architecture: transformer decoder, grouped-query attention (GQA), 32 layers
- Context length: 8,192 tokens
- Training: pretrained on ~15T tokens

## In This Project

Base model for PPO, GRPO, and DPO fine-tuning. In PPO: also used as backbone for the critic network (with a value head). All experiments use the same base checkpoint for fair comparison.

## Connections
- [[ppo]], [[grpo]], [[dpo]] — fine-tuning targets
- [[critic-network]] — PPO adds a value head to this model

## Key Sources
_(populated on paper ingest)_
```

- [ ] **Step 2: Create gsm8k.md**

```markdown
---
type: entity
tags: [gsm8k, benchmark, math, verifiable-reward]
sources: []
updated: 2026-04-15
---

# GSM8K

Grade School Math benchmark. 8,500 grade-school math word problems with verifiable numeric answers.

## Key Properties
- Size: 8,500 problems (train: 7,473 / test: 1,319)
- Task: multi-step arithmetic word problems
- Answer format: final numeric value
- Verification: exact match after extracting final answer

## In This Project

Primary math benchmark. Used in exps 2.7, 2.8, and 2.9. Verifiable rewards computed by extracting and comparing the final numeric answer. Sparse label condition in exp 2.9 uses 10% of training prompts.

## Connections
- [[verifiable-reward]] — reward computation on this dataset
- [[rlvr]] — paradigm this dataset enables
- [[ppo]], [[grpo]], [[dpo]] — all trained and evaluated here

## Key Sources
_(populated on paper ingest)_
```

- [ ] **Step 3: Create humaneval.md**

```markdown
---
type: entity
tags: [humaneval, benchmark, code, verifiable-reward]
sources: []
updated: 2026-04-15
---

# HumanEval

164 hand-crafted Python programming problems, each with a function signature, docstring, and unit tests.

## Key Properties
- Size: 164 problems
- Task: complete a Python function given its signature and docstring
- Verification: execute generated code against provided unit tests
- Standard metric: pass@k

## In This Project

Secondary code benchmark. Used in exp 2.7 (head-to-head across PPO/GRPO/DPO). Verifiable rewards from code execution (pass@1 used as RL training signal).

## Connections
- [[verifiable-reward]] — reward from code execution
- [[rlvr]] — paradigm this dataset enables
- [[ppo]], [[grpo]], [[dpo]] — evaluated here in exp 2.7

## Key Sources
_(populated on paper ingest)_
```

- [ ] **Step 4: Verify entity pages**

```bash
ls RLVR_VAULT/wiki/entities/
```

Expected: `gsm8k.md  humaneval.md  llama-3-8b.md`

- [ ] **Step 5: Commit**

```bash
git add RLVR_VAULT/wiki/entities/
git commit -m "feat: add seed entity pages (LLaMA-3 8B, GSM8K, HumanEval)"
```

---

### Task 7: Create experiment skeleton pages (3 overview pages)

**Files:**
- Create: `RLVR_VAULT/wiki/experiments/exp-2.7/exp-2.7-overview.md`
- Create: `RLVR_VAULT/wiki/experiments/exp-2.8/exp-2.8-overview.md`
- Create: `RLVR_VAULT/wiki/experiments/exp-2.9/exp-2.9-overview.md`

- [ ] **Step 1: Create exp-2.7-overview.md**

```markdown
---
type: experiment
tags: [exp-2.7, ppo, grpo, dpo, gsm8k, humaneval, head-to-head]
sources: []
updated: 2026-04-15
---

# Exp 2.7 — Head-to-Head: PPO vs GRPO vs DPO

## Goal
Compare all three alignment methods on GSM8K and HumanEval with matched compute across 3 random seeds.

## Setup
- Methods: [[ppo]], [[grpo]], [[dpo]] (all on [[llama-3-8b]])
- Datasets: [[gsm8k]], [[humaneval]]
- Seeds: 3 (matched across methods)
- Compute: matched GPU-hours per method
- Metrics: accuracy, training stability, convergence speed, advantage estimation error

## Aggregate Results
_(populated after runs complete)_

| Method | GSM8K Acc (mean±std) | HumanEval Acc (mean±std) | Convergence Step | Stability |
|--------|----------------------|--------------------------|-----------------|-----------|
| PPO | | | | |
| GRPO | | | | |
| DPO | | | | |

## Per-Run Pages
_(created as runs complete)_

## Key Findings
_(populated after runs complete)_

## Open Questions
- Does GRPO's critic-free advantage estimation hurt accuracy on harder problems (HumanEval)?
- Does DPO's offline training limit adaptability compared to online methods?
- How does training stability compare across methods at matched compute?
```

- [ ] **Step 2: Create exp-2.8-overview.md**

```markdown
---
type: experiment
tags: [exp-2.8, ppo, grpo, critic-sweep, advantage-estimation]
sources: []
updated: 2026-04-15
---

# Exp 2.8 — Critic Size Sweep: PPO Variants vs GRPO

## Goal
Determine whether larger critic networks reduce advantage estimation error enough to justify their compute cost, relative to GRPO's zero-critic baseline.

## Setup
- Methods: PPO with critic sizes {none (=GRPO baseline), small, medium, large}
- Dataset: [[gsm8k]] (primary)
- Key output: crossover plot of Monte Carlo advantage error vs accuracy
- Additional metrics: compute cost, convergence speed

## Critic Size Definitions

| Variant | Architecture | Approx. Params |
|---------|-------------|----------------|
| None (GRPO) | No critic | 0 |
| Small | MLP head on frozen policy | ~10M |
| Medium | Full LM head fine-tuned | ~800M |
| Large | Separate transformer | ~8B |

## Aggregate Results
_(populated after runs complete)_

| Critic Size | MC Advantage Error | GSM8K Accuracy | Compute Cost |
|-------------|-------------------|----------------|-------------|
| None (GRPO) | | | |
| Small | | | |
| Medium | | | |
| Large | | | |

## Per-Run Pages
_(created as runs complete)_

## Key Findings
_(populated after runs complete)_

## Open Questions
- At what critic size does accuracy improvement flatten (crossover point)?
- Is the crossover point compute-budget-dependent?
```

- [ ] **Step 3: Create exp-2.9-overview.md**

```markdown
---
type: experiment
tags: [exp-2.9, grpo, dpo, label-regimes, sparse-labels, noisy-labels]
sources: []
updated: 2026-04-15
---

# Exp 2.9 — Label Regimes: GRPO vs DPO

## Goal
Compare GRPO and DPO robustness under three label availability conditions: full labels, sparse labels (10%), and noisy labels (10% flipped).

## Setup
- Methods: [[grpo]], [[dpo]] (both on [[llama-3-8b]])
- Dataset: [[gsm8k]] (primary)
- Label conditions:
  - **Full:** 100% of training prompts have verifiable reward
  - **Sparse:** Only 10% of prompts have any correct rollout (rest yield reward=0 for all samples)
  - **Noisy:** 10% of reward labels randomly flipped (0→1 or 1→0)
- Metrics: accuracy per condition, training stability, convergence speed

## Aggregate Results
_(populated after runs complete)_

| Method | Full Labels | Sparse (10%) | Noisy (10% flip) |
|--------|------------|--------------|-----------------|
| GRPO | | | |
| DPO | | | |

## Per-Run Pages
_(created as runs complete)_

## Key Findings
_(populated after runs complete)_

## Open Questions
- Does DPO's offline nature make it more or less robust to sparse labels than GRPO's online sampling?
- Under noisy labels, does GRPO's group normalization provide implicit noise robustness?
```

- [ ] **Step 4: Verify experiment pages**

```bash
ls RLVR_VAULT/wiki/experiments/exp-2.7/ && \
ls RLVR_VAULT/wiki/experiments/exp-2.8/ && \
ls RLVR_VAULT/wiki/experiments/exp-2.9/
```

Expected: one `*-overview.md` in each directory.

- [ ] **Step 5: Commit**

```bash
git add RLVR_VAULT/wiki/experiments/
git commit -m "feat: add experiment skeleton pages for exp 2.7, 2.8, 2.9"
```

---

### Task 8: Create index.md

**Files:**
- Create: `RLVR_VAULT/index.md`

- [ ] **Step 1: Create index.md**

```markdown
# RLVR Vault — Index

Content-oriented catalog of all wiki pages. Updated on every ingest and after any significant page change.

Parse log: `grep "^## \[" log.md | tail -10`
Search wiki: `./search.sh "query" [subdir]`

---

## Concepts

- [[ppo]] — Proximal Policy Optimization: clipped surrogate objective, requires learned critic, standard in RLHF
- [[grpo]] — Group Relative Policy Optimization: critic-free, normalizes advantages within a response group
- [[dpo]] — Direct Preference Optimization: supervised objective from preference pairs, no RL required
- [[rlvr]] — Reinforcement Learning from Verifiable Rewards: reward from answer correctness, no reward model
- [[advantage-estimation]] — A(s,a) = Q(s,a) - V(s); GAE in PPO vs group normalization in GRPO
- [[kl-penalty]] — KL(π||π_ref) regularization; prevents policy drift from base model
- [[verifiable-reward]] — Binary reward from checking output correctness (numeric match, code execution)
- [[preference-pairs]] — (prompt, chosen, rejected) triples for DPO; synthetically generated via verifiable rewards
- [[critic-network]] — Value network V(s) for PPO; swept across sizes (none/small/medium/large) in exp 2.8

---

## Entities

- [[llama-3-8b]] — Meta LLaMA-3 8B; base policy model for all three methods in this project
- [[gsm8k]] — Grade School Math benchmark; 8,500 problems with verifiable numeric answers
- [[humaneval]] — 164 Python coding problems; verified by unit test execution; pass@k metric

---

## Experiments

- [[exp-2.7-overview]] — Head-to-head: PPO vs GRPO vs DPO on GSM8K + HumanEval, 3 seeds, matched compute
- [[exp-2.8-overview]] — Critic size sweep: PPO (none/small/medium/large) vs GRPO; crossover plot of error vs accuracy
- [[exp-2.9-overview]] — Label regimes: GRPO vs DPO under full / sparse (10%) / noisy (10% flip) labels

---

## Sources

_(populated on first ingest)_

---

## Outputs

_(populated on first query that produces a reusable artifact)_
```

- [ ] **Step 2: Verify all [[links]] in index.md resolve to real files**

```bash
for name in ppo grpo dpo rlvr advantage-estimation kl-penalty verifiable-reward preference-pairs critic-network; do
  [ -f "RLVR_VAULT/wiki/concepts/$name.md" ] && echo "OK: $name" || echo "MISSING: $name"
done

for name in llama-3-8b gsm8k humaneval; do
  [ -f "RLVR_VAULT/wiki/entities/$name.md" ] && echo "OK: $name" || echo "MISSING: $name"
done

for exp in exp-2.7/exp-2.7-overview exp-2.8/exp-2.8-overview exp-2.9/exp-2.9-overview; do
  [ -f "RLVR_VAULT/wiki/experiments/$exp.md" ] && echo "OK: $exp" || echo "MISSING: $exp"
done
```

Expected: all 15 lines print `OK: ...`

- [ ] **Step 3: Commit**

```bash
git add RLVR_VAULT/index.md
git commit -m "feat: add index.md with all 15 seeded pages catalogued"
```

---

### Task 9: First ingest — migrate Notes.md

Demonstrates the ingest workflow end-to-end and creates the first source page.

**Files:**
- Create: `RLVR_VAULT/raw/notes/project-notes.md` (copy of Notes.md)
- Create: `RLVR_VAULT/wiki/sources/project-notes-initial.md`
- Modify: `RLVR_VAULT/index.md` (add Sources entry)
- Modify: `RLVR_VAULT/log.md` (append ingest entry)

- [ ] **Step 1: Copy Notes.md into raw**

```bash
cp Notes.md RLVR_VAULT/raw/notes/project-notes.md
```

- [ ] **Step 2: Create source summary page**

```markdown
---
type: source
tags: [project-notes, ppo, grpo, dpo, exp-2.7, exp-2.8, exp-2.9, gsm8k, humaneval]
sources: []
updated: 2026-04-15
---

# Project Notes — Initial Planning Notes

**Raw file:** `../../../raw/notes/project-notes.md`
**Date captured:** 2026-04-15

## Summary

Internal planning notes scoping the RLVR comparison project. Defines the three core experiments (2.7, 2.8, 2.9), three alignment methods (PPO, GRPO, DPO), two benchmarks (GSM8K, HumanEval), and task ownership per team member.

## Key Claims
- All three methods built on LLaMA-3 8B for fair comparison
- PPO requires a modular critic; GRPO and DPO do not
- DPO requires a synthetic preference pair pipeline using verifiable rewards
- Exp 2.7: head-to-head, matched compute, 3 seeds — logs accuracy, stability, convergence, advantage estimation error
- Exp 2.8: critic size sweep (none/small/medium/large); generates crossover plot of critic error vs accuracy
- Exp 2.9: GRPO vs DPO under full / sparse (10%) / noisy (10% flipped) label conditions

## Relevance to This Project

Primary project scoping document. Informs experiment page structure, metric tracking requirements, and the DPO preference pair pipeline design.

## Connections
- [[ppo]], [[grpo]], [[dpo]] — the three methods
- [[gsm8k]], [[humaneval]] — the two benchmarks
- [[exp-2.7-overview]], [[exp-2.8-overview]], [[exp-2.9-overview]] — experiment scoping
- [[preference-pairs]] — DPO pipeline requirement called out explicitly
- [[critic-network]] — PPO modular critic requirement
- [[advantage-estimation]] — tracked metric across all experiments
```

Write this to `RLVR_VAULT/wiki/sources/project-notes-initial.md`.

- [ ] **Step 3: Update index.md — add Sources entry**

Replace the Sources section in `RLVR_VAULT/index.md`:

```markdown
## Sources

- [[project-notes-initial]] — Initial project planning notes: experiment scope (2.7/2.8/2.9), methods, benchmarks, task ownership
```

- [ ] **Step 4: Append to log.md**

Add this line at the end of `RLVR_VAULT/log.md`:

```markdown

## [2026-04-15] ingest | Project Notes — Initial Planning Notes
```

- [ ] **Step 5: Verify search works end-to-end**

```bash
cd RLVR_VAULT && bash search.sh "advantage estimation" && cd ..
```

Expected: matches in `wiki/concepts/advantage-estimation.md`, `wiki/concepts/ppo.md`, `wiki/experiments/exp-2.8/exp-2.8-overview.md`, `wiki/sources/project-notes-initial.md`

```bash
cd RLVR_VAULT && bash search.sh "sparse" experiments && cd ..
```

Expected: match in `wiki/experiments/exp-2.9/exp-2.9-overview.md`

- [ ] **Step 6: Commit**

```bash
git add RLVR_VAULT/raw/notes/project-notes.md \
        RLVR_VAULT/wiki/sources/project-notes-initial.md \
        RLVR_VAULT/index.md \
        RLVR_VAULT/log.md
git commit -m "feat: first ingest — migrate project Notes.md into wiki"
```

---

### Task 10: Final verification

- [ ] **Step 1: Count all wiki pages**

```bash
find RLVR_VAULT/wiki -name "*.md" | wc -l
```

Expected: 16 (9 concepts + 3 entities + 3 experiment overviews + 1 source)

- [ ] **Step 2: Verify every wiki page has frontmatter**

```bash
for f in $(find RLVR_VAULT/wiki -name "*.md"); do
  head -1 "$f" | grep -q "^---$" || echo "MISSING FRONTMATTER: $f"
done
```

Expected: no output.

- [ ] **Step 3: Verify index.md has 16 link entries**

```bash
grep -c "\[\[" RLVR_VAULT/index.md
```

Expected: 16

- [ ] **Step 4: Verify log.md has both entries**

```bash
grep "^## \[" RLVR_VAULT/log.md
```

Expected:
```
## [2026-04-15] ingest | Wiki initialization — seed pages and schema
## [2026-04-15] ingest | Project Notes — Initial Planning Notes
```

- [ ] **Step 5: Search smoke test**

```bash
cd RLVR_VAULT && bash search.sh "KL" concepts && cd ..
```

Expected: matches in `kl-penalty.md`, `ppo.md`, `grpo.md`, `dpo.md`

- [ ] **Step 6: Verify nothing uncommitted**

```bash
git status
```

Expected: `nothing to commit, working tree clean`

---

## Self-Review

**Spec coverage:**
- ✅ Directory structure (Task 1)
- ✅ `search.sh` (Task 2)
- ✅ `log.md` (Task 3)
- ✅ `CLAUDE.md` with all 10 required sections (Task 4)
- ✅ Concept pages: PPO, GRPO, DPO, RLVR, advantage-estimation, kl-penalty, verifiable-reward, preference-pairs, critic-network (Task 5)
- ✅ Entity pages: llama-3-8b, gsm8k, humaneval (Task 6)
- ✅ Experiment overview skeletons: exp-2.7, 2.8, 2.9 (Task 7)
- ✅ `index.md` with all seeded pages (Task 8)
- ✅ First ingest example demonstrating full workflow (Task 9)
- ✅ Final verification (Task 10)

**Placeholder scan:** No TBDs or TODOs in required content. `_(populated after runs complete)_` in experiment pages is intentional — those await actual run data.

**Consistency:** All `[[link]]` names in concept pages match actual filenames. `index.md` links verified against real files in Task 8 Step 2. `log.md` format consistent across all entries.
