# protein-agent-tiny

Tiny, competition-only agent for AI4S task 3: protein conformational ensemble generation.

It is intentionally not a general research app. There is no frontend and no cross-domain abstraction. The core loop is:

1. Read `data/problems/1.json`, `2.json`, `3.json`.
2. Load factual long-term memory into `memory_context.md`.
3. Probe CPU/GPU/packages into `environment_report.md`.
4. Retrieve a small OpenAlex literature set into `literature_review.md`.
5. Optionally run bounded `all-in-agents` research-plan, hypothesis, experiment, and reflection iterations in a workspace.
6. Keep the best accepted `solver.py` by an internal validation-aware proxy score.
7. Update factual long-term memory under `memory/`.
8. Run the solver for all problems.
9. Package `output.zip`.
10. Validate submission format.
11. Generate a short technical report.

`solver.py` is the tiny model artifact by default. If a later agent creates trainable weights, it should write them under `outputs/latest/model/`.

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

This does not call an LLM. It gives a valid, bounded submission.

```bash
scripts/run_baseline.sh 1
```

The output archive is:

```text
outputs/latest/output.zip
```

## Agent Improvement Run

This uses `all-in-agents` to run bounded iterations over a workspace copy of the solver, then runs the suite with the best accepted solver.

```bash
scripts/run_agent.sh 2 20 1
```

Arguments are:

```text
scripts/run_agent.sh <agent_iterations> <max_minutes_per_iteration> <solver_candidate_rounds>
```

Each agent iteration loads the workspace skill `.skills/protein-ensemble/SKILL.md`, reads `memory_context.md`, `environment_report.md`, `literature_review.md`, and `iteration_context.json`, then writes `research_plan.md` before `hypothesis.md`. The plan lets the agent choose its mode for that iteration: literature review, environment setup, dependency experiment, modeling, scoring analysis, code evolution, or observation-only audit.

The workspace also contains `requirements-agent.txt` and a copy of `pyproject.toml`. The agent may edit those files and install small public dependencies with `python -m pip install -r requirements-agent.txt`, but `solver.py` must keep a bounded fallback if optional imports fail. Observation-only iterations are allowed when justified; responses that stop at `max_tokens` or omit `research_plan.md`/`hypothesis.md` are rejected.

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
.venv/bin/python -m protein_agent_tiny.run_suite --clean --rounds 1
.venv/bin/python -m protein_agent_tiny.agent_runner --iterations 2 --max-minutes 20 --solver-rounds 1
.venv/bin/python -m protein_agent_tiny.validate --submission-dir outputs/latest/submission
```
