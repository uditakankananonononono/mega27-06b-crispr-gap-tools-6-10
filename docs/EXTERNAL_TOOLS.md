# External tools, packages, databases, and web resources used (honest inventory)

Legend: RUN = executed in our pipeline; IMPL = re-implemented from the
publication and used for verification; REF = used as published reference
numbers/verification source; DATA = public dataset/data source consumed.

| # | External tool/resource | Type | Use | How |
|---|---|---|---|---|
| 1 | ViennaRNA 2.7.2 (RNAduplex/duplexfold, RNAfold) | package | Tool 6 duplex scoring, self-fold terms | RUN |
| 2 | PyTorch 2.14 (cpu) | package | GuideCNN, PairCNN, SequenceGCN | RUN |
| 3 | scikit-learn | package | GBT/HistGBT/ridge models, metrics | RUN |
| 4 | Biopython | package | RefSeq fetch, seq utilities | RUN |
| 5 | NumPy | package | all numerics | RUN |
| 6 | pandas | package | all data handling | RUN |
| 7 | SciPy | package | Spearman, stats tests | RUN |
| 8 | matplotlib | package | all figures | RUN |
| 9 | pytest | package | 40-test hermetic suite | RUN |
| 10 | NCBI E-utilities (esearch/efetch) | web API | RefSeq accession resolution + sequence (Xrcc4, Piga, Cd9) | RUN |
| 11 | inDelphi U2OS Lib-A event table (figshare 6837956) | database | Tool 7 training/eval | DATA+RUN |
| 12 | inDelphi targets-libA (maxwshen GitHub) | database | Tool 7 target contexts | DATA+RUN |
| 13 | Doench FC+RES (Azimuth mirror) | dataset | Tool 10 training | DATA+RUN |
| 14 | Doench V1 suppl (mouse) | dataset | Tool 10 evaluation | DATA+RUN |
| 15 | crisprSQL 100720 | database | off-target baselines/provenance | DATA+RUN |
| 16 | Kosicki 2018 (nbt.4192) loci | publication data | Tool 8 validation loci | REF+DATA |
| 17 | Azimuth / Rule Set 2 (bioRxiv 021568) | published tool | benchmark leader; features+hyperparams replicated | IMPL+REF |
| 18 | Common Mechanism (ibbis-bio) | published tool | Tool 9 wrap target; audit format | REF (deploy-step documented) |
| 19 | SantaLucia 1998 NN thermodynamics | published model | thermo_dg features | IMPL |
| 20 | GitHub | platform | repo + CI surface | RUN |
| 21 | pdfLaTeX (TeX Live) | package | paper build | RUN |
| 22 | Hsu 2013 MIT off-target score | published tool | crisprSQL baseline comparisons | REF |
| 23 | Doench 2016 CFD score | published tool | crisprSQL baseline comparisons | REF |
| 24 | Lindel (Chen 2019) | published tool | Tool 7 reference numbers | REF |
| 25 | inDelphi model (Shen 2018) | published tool | Tool 7 reference numbers | REF |

Current honest count: 25 external tools/resources (11 RUN, 5 DATA+RUN,
3 IMPL/IMPL+REF, 6 REF). Target: 40. Gap plan: additional public datasets
(Lindel training set, Wang/Koike-Yusa viability, Chari 2015, CRISPRscan
zebrafish, inDelphi additional cell lines) and verification implementations
of published scorers (Rule Set 1, SSC, CRISPRscan) benchmarked in the repo.

## Datasets consumed (honest count)
1. inDelphi U2OS Lib-A events (117,438 rows)
2. inDelphi targets-libA (2,000 targets)
3. Doench FC+RES (5,310 guides)
4. Doench V1 mouse (2,144 guides)
5. crisprSQL 100720 (25,632 records)
6. Xrcc4 NM_028012.4 (RefSeq)
7. Piga NM_011081.4 (RefSeq)
8. Cd9 NM_007657.4 (RefSeq)

Current honest count: 8 datasets/data sources. Target: 120+.
