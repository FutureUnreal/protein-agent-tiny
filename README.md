# protein-agent-tiny

Tiny, competition-only agent for AI4S task 3: protein conformational ensemble generation.

It is intentionally not a general research app. There is no frontend and no cross-domain abstraction. The core loop is:

1. Read `data/problems/1.json`, `2.json`, `3.json`.
2. Let a small `all-in-agents` agent optionally improve `protein_agent_tiny/solver.py`.
3. Run the solver for all problems.
4. Package `output.zip`.
5. Validate submission format.
6. Generate a short technical report.

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
scripts/deploy_uv.sh
```

Set `OPENAI_API_KEY`, `OPENAI_API_BASE`, and `PROTEIN_AGENT_MODEL` in `.env` or shell environment.

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

This uses `all-in-agents` to edit a workspace copy of `solver.py`, then runs the suite.

```bash
scripts/run_agent.sh 2 20
```

The latest agent workspace is under `workspaces/`, and the latest packaged submission remains under `outputs/latest/output.zip`.

## Technical Report

```bash
.venv/bin/python -m protein_agent_tiny.report --run-dir outputs/latest
```

This writes `outputs/latest/technical_report.md`.

## Direct Module Commands

The uv deployment installs the package into `.venv`, so these are equivalent:

```bash
.venv/bin/python -m protein_agent_tiny.run_suite --clean --rounds 1
.venv/bin/python -m protein_agent_tiny.agent_runner --rounds 2 --max-minutes 20
.venv/bin/python -m protein_agent_tiny.validate --submission-dir outputs/latest/submission
```
