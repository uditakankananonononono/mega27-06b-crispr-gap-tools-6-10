# mega27-06b-crispr-gap-tools-6-10

Item 6 (second half) of MEGA-PROGRAM-27: five CRISPR/biohacking gap tools
(gaps 6-10 of the verified 10-gap audit), each on real public data with
hermetic tests and honest benchmarks.

## Tools

| Gap | Tool | Package | Headline result |
|-----|------|---------|-----------------|
| 6 | Multiplex cross-guide interference screener | `tools_multiplex/` | ViennaRNA duplex + PAM-competition decay + GNN removal ranking (vs brute-force ground truth); CNN/GCN duplex surrogates at honest parity with ridge (0.63-0.65 Spearman vs ViennaRNA teacher) |
| 7 | Repair-outcome annotator | `tools_outcome/` | Real inDelphi U2OS LibA (1,742 sites): +1-insertion Spearman 0.60 (GBT), frameshift 0.31, MH-deletion 0.17 - below published per-genotype models, reported as-is |
| 8 | Cut-site large-deletion risk flagger | `tools_riskflag/` | Mechanistic (direct-repeat spans, complexity) per Kosicki 2018 + Wen 2022; all three Kosicki loci (real RefSeq Xrcc4/Piga/Cd9) flag HIGH/MODERATE |
| 9 | Biosecurity screen-and-design wrapper | `tools_biosecurity/` | commec (IBBIS Common Mechanism) backend, blocking + audit; honest UNAVAILABLE when engine absent |
| 10 | Cross-species transfer evaluator | `tools_transfer/` | CNN human (Doench FC+RES) -> mouse (V1): Spearman 0.78 vs 0.63 held-out - no mammalian transfer penalty |

## Layout

- `crisprlib/` - shared core: featurization, GuideCNN/PairCNN, pure-torch GCN on
  k-mer transition graphs, deterministic training loop, MIT-score (Hsu 2013) baseline.
- `tools_*/` - one package per gap.
- `scripts/` - live benchmark/validation scripts (network + training; not CI).
- `results/` - benchmark JSON outputs from the live runs.
- `data/` - public datasets (see data/PROVENANCE.md, incl. blocked sources).
- `paper/main.tex` - manuscript (pdflatex, Times); figures from scripts/make_figures.py.
- `cli.py` - unified CLI: multiplex / riskflag / transfer / bioscreen.

## Tests

`python3 -m pytest tests/ -q` - 40 tests, hermetic (no network, no training).

## Live runs (outside CI)

```
python3 scripts/benchmark_duplex_surrogate.py
python3 scripts/train_outcome_model.py
python3 scripts/benchmark_transfer.py
python3 scripts/validate_riskflag_realseq.py
python3 scripts/make_figures.py
```
