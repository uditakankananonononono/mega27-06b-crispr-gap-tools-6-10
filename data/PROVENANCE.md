# Data provenance (all public, no auth)

- FC_plus_RES_withPredictions.csv - Doench 2016 (Nat Biotechnol 34:184) human guide
  activity set, mirrored from MicrosoftResearch/azimuth repo (5,310 guides). Via lane C.
- V1_suppl_data.txt - Doench 2016 V1 mouse activity set (2,144 guides), same mirror. Via lane C.
- U2OS_LibA_postCas9_rep1.csv - Shen et al. 2018 (Nature 563:646) inDelphi event counts,
  figshare 6837956 file 13502321.
- targets-libA.txt, names-libA.txt - inDelphi LibA 57-nt target contexts,
  maxwshen/inDelphi-dataprocessinganalysis data-libprocessing.
- crisprsql/100720.csv - crisprSQL cleavage database dump (www.crisprsql.com/downloads/100720.zip),
  25,632 guide-target records incl. epigenetic features, genomes hg19/hg38/mm10/mm9/rn5.

## Cross-species data sources tried and blocked (documented boundary)
- CRISPRz (zebrafish, 505 validated guides): site down (crisprz.stanford.edu unreachable).
- Varshney 2015 (Genome Res 25:1030) zebrafish supp tables: EuropePMC supplementary bundle
  contains figures only; CSHL DC1 link 404s.
- Arabidopsis lignin-screen sgRNA set (figshare collection 4515980): supplements are PDFs
  (no machine-readable table); PDF table extraction attempted only if time permits.
