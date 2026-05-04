# Protein Ensemble Competition Skill

Use this skill when improving `solver_pkg/` for AI4S task 3.

The only official inputs are `problems/1.json`, `problems/2.json`, and `problems/3.json`.
Each contains a single-chain amino-acid sequence and a reference conformer count.

Required output:
- mmCIF files named `{problem_id}_conf{N}_pred.cif`
- at most 10 conformers per problem
- finite single-chain coordinates with `_atom_site` and CA atoms
- `final_info.json` with diversity and validation-oriented metrics

Data governance:
- Do not use this competition's original MD trajectories, crystal structures, or NMR ensembles.
- Do NOT perform sequence-similarity or template searches (BLAST, PSI-BLAST, HMMER, MMseqs2, Foldseek) using the competition sequences. Selecting structural/evolutionary homologs of the competition proteins from PDB or AFDB counts as indirect use of target-derived data and is forbidden.
- Public pretrained models, public force fields, unrelated PDB entries (different fold class), AlphaFold DB,
  UniProt/UniRef/MGnify, and unrelated public MD benchmark datasets are allowed, but
  optional.
- **Fail loudly, never substitute placeholders.** If an optional resource fails (missing package, network unreachable, model weights unavailable), exit non-zero with a clear error. Per-problem zero scores are acceptable (FAQ Q4); fabricated geometric placeholders are not.

Iteration protocol:
1. Read `iteration_context.json`.
2. Read `memory_context.md`, `environment_report.md`, and `literature_review.md`.
3. Write `research_plan.md` first. Choose the iteration mode yourself: literature review,
   environment setup, dependency experiment, modeling, scoring analysis, code evolution,
   or observation-only audit.
4. Cite one relevant prior fact, environment constraint, or literature implication in `hypothesis.md`.
5. Write a concise `hypothesis.md` with at most 12 bullet lines.
6. If the hypothesis requires implementation, edit `solver_pkg/`; observation-only iterations are allowed when justified.
7. Run a bounded smoke test before finishing when code changed or dependencies changed.
8. Append concise evidence to `notes.md`.
