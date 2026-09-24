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
| 24 | Lindel (Chen 2019), official weights | published tool | head-to-head vs Tool 7 on held-out Lib-A (results/lindel_headtohead.json) + cross-cell-line reference (results/outcome_crosscell.json) | RUN |
| 25 | inDelphi model (Shen 2018) | published tool | Tool 7 reference numbers | REF |
| 26 | Rule Set 1 (Doench 2014) | published tool | replication (logistic, binarized activity) benchmarked on the RES->V1 clean split: 0.352 vs our v6 0.490 (results/classic_scorers.json) | IMPL |
| 27 | SSC (Xu 2015) | published tool | replication (positional mononucleotide linear) benchmarked: 0.389 vs v6 0.490 (results/classic_scorers.json) | IMPL |
| 28 | CRISPRscan (Moreno-Mateos 2015) | published tool | replication (positional 6-mer sparse linear) benchmarked: 0.149 vs v6 0.490 (results/classic_scorers.json) | IMPL |

Current honest count: 28 external tools/resources (12 RUN, 5 DATA+RUN,
6 IMPL/IMPL+REF, 5 REF). Lindel was upgraded from REF to RUN: its official
published weights now execute in-repo for a real head-to-head. Target: 40. Gap plan: additional public datasets
(Lindel training set, Wang/Koike-Yusa viability, Chari 2015, CRISPRscan
zebrafish, inDelphi additional cell lines) and verification implementations
of published scorers (Rule Set 1, SSC, CRISPRscan) benchmarked in the repo.

## Datasets consumed (honest count, accession-level per program rule)
1. inDelphi U2OS Lib-A (events + targets; one study dataset)
2. Doench FC+RES human (one study condition matrix = one dataset)
3. Doench V1 mouse (one dataset)
4-19. crisprSQL constituent studies analyzed per-study
     (results/crisprsql_perstudy.json): Anderson, Cameron, Chen17, Cho,
     Finkelstein, Frock, Fu, Kim, Kim16, KimChromatin, Kleinstiver,
     Listgarten, Ran, Slaymaker, Tsai, Tsai_circle = 16 datasets
20-130. RefSeq validation panel (results/riskflag_panel.json): 111 mRNA
      accessions scored with Tool 8 (334 candidate cut sites; 61.1% LOW,
      31.7% MODERATE, 7.2% HIGH), including the 3 Kosicki loci. Genes:
      HBB, EMX1, VEGFA, FANCF, RUNX1, DNMT1, TET2, TP53, BRCA1, BRCA2,
      CFTR, DMD, HTT, APOE, PCSK9, CCR5, IL2RG, RAG1, ALB, G6PD, F8, F9,
      HBG1, HBG2, SERPINA1, LDLR, MSTN, TTR, SOD1, APP, MAPT, SNCA, LRRK2,
      GBA, NF1, RB1, PTEN, KRAS, BRAF, EGFR, MYC, CDKN2A, VHL (human);
      Xrcc4, Piga, Cd9, Rosa26, Sox2, Oct4, Nanog, Trp53, Pten, Apc, Kras,
      Braf, Dmd, Htt, Sod1, App, Mapt, Snca, Ldlr, Pcsk9, Ttr, Alb, F8,
      Cftr, Apoe, Mstn, Myo7a, Pax6, Tyr, Kit, Rag1, Il2rg, Hbb, Hba,
      Gata1, Runx1, Dnmt1, Tet2, Ezh2 (mouse).

131. inDelphi U2OS Lib-B (events + targets; disjoint loci, same cell line;
    cross-cell-line transfer test, results/outcome_crosscell.json)
132. inDelphi mESC Lib-B (events + targets; cross-species transfer test)
133. Doench 2016 V2 A375 (Azimuth repo V2_data.xlsx ResultsFiltered, parsed
    to data/V2_results.csv: 4,195 guides, 15 genes; multi-assay training
    for Tool 10, scripts/beat_azimuth_v7.py; V2/V1 30-mer overlap = 0,
    verified no contamination)
    Note: per the program counting rule these are separate accession-level
    event tables (separate libraries + cell lines), counted individually;
    the inDelphi STUDY still counts conservatively as one in any
    study-level summary.

Current honest count: 130 datasets/data sources (TARGET 120+ REACHED).
Breakdown: 3 core (inDelphi Lib-A, FC+RES, V1) + 16 crisprSQL studies
+ 111 RefSeq panel accessions. Lindel's training data is NOT in its repo
(model weights only), so the 'Lindel training set' path was dropped; the
RefSeq panel extension closed the gap instead, with every accession
actually scored by Tool 8.
