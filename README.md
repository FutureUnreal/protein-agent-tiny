# protein-agent-tiny

Tiny, competition-only agent for AI4S task 3: protein conformational ensemble generation.

It is intentionally not a general research app. There is no frontend and no cross-domain abstraction. The core loop is:

1. Read official-schema `data/problems/1.json`, `2.json`, `3.json` files. Each file is a single-record list with `name` (`r001` etc.), `sequences[0].proteinChain.sequence`, and `conformer_count`; output filenames still use problem IDs `1`, `2`, `3`.
2. Load factual long-term memory into `memory_context.md`.
3. Probe CPU/GPU/packages into `environment_report.md`.
4. Retrieve a small OpenAlex literature set into `literature_review.md`.
5. Route to bootstrap-agent or improve-agent based on presence of `solver_pkg/.pipeline_ready` sentinel.
6. Keep the best accepted `solver_pkg/` snapshot in `best_pipeline/` by a generated-CIF internal selection score, and retain every scored candidate under `candidate_pipelines/`.
7. Update factual long-term memory under `memory/`.
8. Run the pipeline CLI for all problems.
9. Package `output.zip`.
10. Validate submission format.
11. Generate a short technical report.

## Agent Architecture

The agent runs as a state machine with three roles:

1. **bootstrap-agent** — runs ONCE per workspace when `solver_pkg/.pipeline_ready` is absent.
   System prompt declares "you are creating the FIRST VERSION of a protein conformational
   ensemble pipeline FOR THIS WORKSPACE". It must produce `solver_pkg/cli.py`,
   `solver_pkg/pipeline.py`, `research_plan.md`, `hypothesis.md`, and `notes.md`.
   The runtime validates the CLI with a smoke test and writes the sentinel. After
   `PROTEIN_AGENT_BOOTSTRAP_MAX_ATTEMPTS` (default 2) consecutive failures, the runtime
   **records the failure honestly** in `agent.log` (`mode=bootstrap_failed`) and produces
   a submission with **zero CIFs**. No placeholder geometry is fabricated.

2. **improve-agent** — runs every iteration once `solver_pkg/.pipeline_ready` exists.
   System prompt declares "you are evolving an EXISTING pipeline located at solver_pkg/".
   Each iteration writes `research_plan.md` and `hypothesis.md`, makes a minimal bounded
   change to `solver_pkg/`, runs a smoke test, and updates an `accepted_history`. The
   `best_pipeline/` directory holds the snapshot accepted by hard-gate + internal
   selection score; `candidate_pipelines/` keeps the accepted and rejected scored
   snapshots so later iterations can compare branches instead of only chasing the
   latest scalar.

3. **reflect-agent** — runs after every improve iteration with no tools and a single LLM
   call. Produces `observation_<iter>.md` with Evidence / Supported / Risks / Open Questions.

### Failure semantics (honest, not fabricated)

The repo has exactly one path for CIFs to reach `output.zip`: a real `solver_pkg/cli.py`
written by the agent and executed via `run_suite`. There is no geometric fallback.

- Bootstrap fails → submission contains `agent.log` only; per-problem scores will be 0.
  FAQ Q4 explicitly allows this. The agent.log carries the real error so the next run
  can diagnose it.
- Improve solver crashes on some problem → whatever CIFs were produced before the crash
  are kept; the rest are absent. `agent.log` records `final_run_error`.
- `--skip-agent` is a **package-only audit mode** for CI: zero CIFs, `audit_only=true`
  in agent.log. Not a valid competition submission.

### Why no human-written `solver.py` in the repo

This repo intentionally does NOT track any scientific baseline. The agent's job
includes producing the first version. The only human-authored Python the agent
inherits is:

- `protein_agent_tiny/prompts/*.md` — behavior instructions.
- `protein_agent_tiny/runtime/*.py` — the state-machine runtime itself (orchestration only).

`protein_agent_tiny/solver.py` and `protein_agent_tiny/solver_pkg/` are explicitly
in `.gitignore` to prevent agent-produced artifacts from leaking into the repo.

## Module Boundaries

High-cohesion / low-coupling rules enforced by import DAG tests:

- `scoring/` only depends on stdlib + numpy; no runtime, no LLM dependencies.
- `runtime/` depends on `scoring/` + `prompts/` + `tools/` (one-way); never the reverse.
- `agent_runner.py` is ≤120 lines, only orchestration; imports only `runtime.contracts` and `runtime.iteration`.
- `prompts/` is markdown only; a single `__init__.py` provides `load(name)`.

## Rules Encoded

- Required archive: `output.zip`
- Required files: `{problem_id}_conf{N}_pred.cif` and `agent.log`
- Official problem IDs: `1`, `2`, `3`
- At most 10 conformers per problem
- mmCIF with `_atom_site` and finite coordinates
- Single protein chain
- Must use sequence as input
- Forbidden input: this competition's original MD trajectories, crystal structures, or NMR ensembles
- Allowed optional resources: public pretrained models, public force fields, public protein databases such as RCSB PDB, AlphaFold DB, UniProt/UniRef/MGnify, and unrelated public MD benchmark datasets
- Three-agent routing: bootstrap-agent → improve-agent (each followed by reflect-agent)
- `proxy_score` is computed from real CIF coordinates via `scoring/proxy.py`; it is never read from `final_info.json` self-reports
- Hard gate covers: missing CA atoms, non-finite coordinates, CA-CA spacing deviation >0.15Å from 3.8Å, severe CA-CA clash (<3.0Å non-adjacent), implausible radius of gyration, all-conformers-duplicate ensemble

## Setup

```bash
cd protein-agent-tiny
bash scripts/bootstrap_server.sh
```

Set `OPENAI_API_KEY`, `OPENAI_API_BASE`, and `PROTEIN_AGENT_MODEL` in `.env` or shell environment. `OPENALEX_API_KEY` is optional for literature retrieval.
For large-context models, `PROTEIN_AGENT_MAX_INPUT_TOKENS` and `PROTEIN_AGENT_MAX_OUTPUT_TOKENS` control the per-call budget passed to `all-in-agents`.

`bootstrap_server.sh` is the server-friendly first-run command. It installs `uv` if needed, syncs `.venv`, creates `.env` only when missing, runs a baseline smoke test, validates `outputs/latest/submission`, and prints the next agent command. To skip the baseline smoke test:

```bash
RUN_BASELINE=0 bash scripts/bootstrap_server.sh
```

For day-to-day dependency sync without a smoke test:

```bash
bash scripts/deploy_uv.sh
```

## Fast Baseline

This does not call an LLM. It runs `--skip-agent`, which produces a **package-only audit submission** (zero CIFs, agent.log marked `audit_only=true`). Use this only to verify packaging/validation plumbing; it is NOT a valid competition submission.

```bash
scripts/run_baseline.sh 1
```

Internally this runs:

```bash
python -m protein_agent_tiny.agent_runner --skip-agent --iterations 1
```

The `--skip-agent` flag bypasses the LLM bootstrap/improve/reflect loop entirely. It writes a complete `agent.log` (5 required event types, all marked `audit_only=true, mode=package_only`) and **zero CIFs**. The packaging/validation pipeline still runs, so this is the canonical CI smoke test.

The output archive is:

```text
outputs/latest/output.zip
```

## Agent Improvement Run

This uses `all-in-agents` to run bounded iterations, routing to bootstrap-agent on first run and improve-agent on subsequent runs.

```bash
scripts/run_agent.sh 2 20 1
```

Arguments are:

```text
scripts/run_agent.sh <agent_iterations> <max_minutes_per_iteration> <solver_candidate_rounds>
```

Each agent iteration loads the workspace skill `.skills/protein-ensemble/SKILL.md`, reads `memory_context.md`, `environment_report.md`, `literature_review.md`, and `iteration_context.json`, then writes `research_plan.md` before `hypothesis.md`. The plan lets the agent choose its mode for that iteration: literature review, environment setup, dependency experiment, modeling, scoring analysis, code evolution, or observation-only audit.

If no local system Python with torch is available, the runtime creates an agent-owned `pyproject.toml` under the current workspace and syncs it with uv. The agent may edit that workspace manifest when public dependencies are justified by `environment_report.md`; it must not edit the project root dependency files. Observation-only iterations are allowed when justified; responses that stop at `max_tokens` or omit `research_plan.md`/`hypothesis.md` are rejected.

The latest agent workspace is under `workspaces/`, factual memory is under `memory/`, and the latest packaged submission remains under `outputs/latest/output.zip`.
Every baseline or agent run also writes a timestamped snapshot under `outputs/archive/` so a later run does not overwrite the last usable package.

The runner uses an internal proxy score only for local selection. Experiments with missing outputs, nonfinite coordinates, no conformers, extreme radius of gyration, extreme pairwise RMSD, or gross CA spacing violations are hard-rejected before proxy comparison.

## Official Score Feedback

After submitting `output.zip`, record the official result so future agent runs can see real feedback in `memory_context.md`:

```bash
bash scripts/record_score.sh --score 0.85 --score1 0.87 --score2 0.88 --success true --notes "leaderboard trial"
```

Or paste the raw official JSON:

```bash
bash scripts/record_score.sh --score-json '{"score":0.85,"scoreJson":{"score1":0.87,"score2":0.88},"success":true,"errorMsg":""}'
```

On shells where JSON quoting is awkward, put the official response in a file:

```bash
bash scripts/record_score.sh --score-json-file official_score.json
```

## Memory

Generated memory is local runtime state and is intentionally ignored by git. It records factual observations, not prescribed next steps:

```text
memory/
|-- observations.md
|-- runs.jsonl
|-- best_runs.jsonl
|-- scores.jsonl
`-- environment_report.json
```

## Technical Report

```bash
.venv/bin/python -m protein_agent_tiny.report --run-dir outputs/latest
```

This writes `outputs/latest/technical_report.md`.

## Direct Module Commands

The uv deployment installs the package into `.venv`, so these are equivalent:

```bash
.venv/bin/python -m protein_agent_tiny.agent_runner --skip-agent --iterations 1
.venv/bin/python -m protein_agent_tiny.agent_runner --iterations 2 --max-minutes 20 --solver-rounds 1
.venv/bin/python -m protein_agent_tiny.validate --submission-dir outputs/latest/submission
```
