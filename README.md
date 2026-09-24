# mega27-06b-crispr-gap-tools-6-10

Item 6 (second half) of the MEGA-PROGRAM-27 computational-biology program: five CRISPR/biohacking
gap tools, each with CNN and GNN scoring models, trained on real public datasets and benchmarked
against published leaders (DeepCRISPR/CRISPRoff-class off-target, PRIDICT/DeepPrime prime-editing,
BE-Hive/DeepBE base-editing, Lindel/inDelphi/FORECasT repair-outcome, Doench Rule Set 2/CRISPRon
on-target baselines, per assigned gap).

Shared core (`crisprlib/`): sequence featurization, CNN/GNN model components, dataset loading and
benchmark harness. Tool packages land under `tools_<gapname>/` as gaps are assigned from the
verified 10-gap list.

Rules: real open datasets only; hermetic pytest suite (network only in dataset-fetch scripts and
explicitly-marked live tests); honest benchmark verdicts.
